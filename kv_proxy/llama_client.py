"""Async HTTP client for llama-swap (and, optionally, direct llama-server).

llama-swap proxies the OpenAI endpoints but NOT /slots. Slot operations go
through its /upstream/{model}/... passthrough (or a configured direct upstream
host:port as an escape hatch).
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import httpx

log = logging.getLogger("kv_proxy.llama")


class LlamaError(Exception):
    """Upstream returned an error for a slot operation."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"upstream {status}: {detail}")
        self.status = status
        self.detail = detail


class LlamaClient:
    def __init__(self, swap_url: str, direct_upstream: dict[str, str] | None = None,
                 timeout: float = 30.0, direct_llama_server: bool = False,
                 router_upstream: str = "", router_url: str = "",
                 upstream_api_key: str = ""):
        self.swap_url = swap_url.rstrip("/")
        self.direct_upstream = {k: v.rstrip("/") for k, v in (direct_upstream or {}).items()}
        # direct_llama_server: upstream is a plain llama-server, not llama-swap.
        # Slot ops then hit /slots directly; chat requests still use the body's
        # model field (llama-server accepts and ignores it).
        self.direct_llama_server = direct_llama_server
        # router_upstream: llama-swap's upstream entry hosts a llama-server
        # router (--models-preset). Slot ops go through /upstream/{router}/...
        # carrying the *router-level* model name in the query (GET) or body
        # (POST) — the router reads it there and forwards to the right child.
        # Model residency comes from the router's /models, polled directly so
        # the poll never wakes llama-swap's upstream (ttl stays meaningful).
        self.router_upstream = router_upstream
        self.router_url = router_url.rstrip("/") if router_url else ""
        self.upstream_api_key = upstream_api_key
        self._client = httpx.AsyncClient(
            base_url=self.swap_url,
            timeout=httpx.Timeout(timeout, read=3600.0),  # long reads: generations
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------- slot ops

    def _auth_headers(self) -> dict[str, str]:
        """Auth for kv-proxy's own control-plane calls.

        llama-swap protects /upstream with its apiKeys (server.go wires the
        auth middleware into upstreamChain), so slot ops need the key even
        though chat forwards the client's own Authorization header.
        """
        if self.upstream_api_key:
            return {"Authorization": f"Bearer {self.upstream_api_key}"}
        return {}

    def _router_model(self, model: str, body: dict | None = None,
                      params: dict | None = None) -> tuple[dict, dict]:
        """Apply router-mode model routing to a slot op.

        Router proxy_post reads `model` from the JSON body (server-models.cpp
        proxy_post), proxy_get from the query string — so GET carries it as a
        query param and POST embeds it in the body. Child servers ignore the
        extra field, so the same call works without a router too.
        """
        body = dict(body) if body is not None else {}
        params = dict(params) if params is not None else {}
        if self.router_upstream:
            body.setdefault("model", model)
            params.setdefault("model", model)
        return body, params

    def _slot_path(self, model: str, sub: str = "") -> str:
        """Path (relative to base_url) for /slots operations.

        llama-swap mode:  /upstream/{model}/slots[/sub]
        router mode:      /upstream/{router_upstream}/slots[/sub]
        direct mode:      /slots[/sub]
        per-model override via [direct_upstream]: absolute host:port + /slots
        """
        if self.direct_llama_server:
            return f"/slots{('/' + sub) if sub else ''}"
        upstream = self.router_upstream or model
        return f"/upstream/{upstream}/slots{('/' + sub) if sub else ''}"

    async def get_slots(self, model: str) -> list[dict]:
        params = {"model": model} if self.router_upstream else None
        r = await self._client.get(
            self._slot_path(model), params=params, headers=self._auth_headers()
        )
        if r.status_code != 200:
            raise LlamaError(r.status_code, r.text[:300])
        return r.json()

    async def save_slot(self, model: str, slot_id: int, filename: str) -> dict:
        body, params = self._router_model(model, {"filename": filename},
                                          {"action": "save"})
        r = await self._client.post(
            self._slot_path(model, f"{slot_id}"),
            params=params,
            json=body,
            headers=self._auth_headers(),
        )
        if r.status_code != 200:
            raise LlamaError(r.status_code, r.text[:300])
        return r.json()

    async def restore_slot(self, model: str, slot_id: int, filename: str) -> dict:
        body, params = self._router_model(model, {"filename": filename},
                                          {"action": "restore"})
        r = await self._client.post(
            self._slot_path(model, f"{slot_id}"),
            params=params,
            json=body,
            headers=self._auth_headers(),
        )
        if r.status_code != 200:
            raise LlamaError(r.status_code, r.text[:300])
        return r.json()

    async def erase_slot(self, model: str, slot_id: int) -> dict:
        # a JSON body is mandatory in router mode: the router's proxy_post
        # json-parses every POST body before it can route it
        body, params = self._router_model(model, {}, {"action": "erase"})
        r = await self._client.post(
            self._slot_path(model, f"{slot_id}"),
            params=params,
            json=body or None,  # no empty-{} body against plain llama-server
            headers=self._auth_headers(),
        )
        if r.status_code != 200:
            raise LlamaError(r.status_code, r.text[:300])
        return r.json()

    # ------------------------------------------------------------- discovery

    async def running_models(self) -> list[str]:
        """Model ids currently resident (KV reachable without a reload).

        llama-swap: GET /running. Router mode: the router's /models, polled
        directly at router_url — llama-swap only ever reports the router
        itself, never its children. Anything the router keeps in memory
        (loaded, loading, sleeping) counts; 'unloaded' means the KV is gone.
        Direct llama-server: the single model it serves (read from /v1/models).
        """
        if self.router_upstream:
            try:
                r = await self._client.get(
                    f"{self.router_url}/models",
                    headers=self._auth_headers(),
                )
                if r.status_code != 200:
                    return []
                data = r.json()
            except (httpx.HTTPError, ValueError):
                return []
            items = data if isinstance(data, list) else data.get("data", [])
            out = []
            for item in items if isinstance(items, list) else []:
                if not isinstance(item, dict):
                    continue
                status = item.get("status")
                value = status.get("value") if isinstance(status, dict) else status
                # 'loaded': KV resident. 'sleeping': the child process is
                # frozen by the router's --sleep-idle-seconds — verified in our
                # build (server-queue.cpp should_sleep) that sleep only pauses
                # the task loop and does NOT free the KV cache. 'loading' is
                # deliberately excluded: no resident KV yet — a child that
                # restarted between poll ticks must be invalidated, and
                # skipping that window would treat empty slots as resident.
                if value in ("loaded", "sleeping"):
                    mid = item.get("id")
                    if mid:
                        out.append(str(mid))
            return out
        if self.direct_llama_server:
            try:
                data = await self.get_models()
                ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
                return [str(i) for i in ids if i]
            except (httpx.HTTPError, ValueError):
                return []
        try:
            r = await self._client.get("/running")
            if r.status_code != 200:
                return []
            data = r.json()
        except (httpx.HTTPError, ValueError):
            return []
        if isinstance(data, list):
            out = []
            for item in data:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    mid = item.get("id") or item.get("model") or item.get("name")
                    if mid:
                        out.append(str(mid))
            return out
        if isinstance(data, dict):
            # tolerate {"models": [...]} or {model_id: {...}} shapes
            if "models" in data and isinstance(data["models"], list):
                return [str(m) for m in data["models"]]
            return list(data.keys())
        return []

    async def is_resident(self, model: str) -> bool | None:
        """Is `model` resident (KV reachable without a load)?

        True/False when residency is *known*; None when the residency poll
        itself failed (caller must not act on an unknown state). Never
        triggers a load: router mode polls router_url/models directly,
        llama-swap mode GET /running, direct mode always True.
        """
        if self.router_upstream:
            try:
                r = await self._client.get(
                    f"{self.router_url}/models",
                    headers=self._auth_headers(),
                )
                if r.status_code != 200:
                    return None
                data = r.json()
            except (httpx.HTTPError, ValueError):
                return None
            items = data if isinstance(data, list) else data.get("data", [])
            for item in items if isinstance(items, list) else []:
                if not isinstance(item, dict):
                    continue
                if str(item.get("id") or "") != model:
                    continue
                status = item.get("status")
                value = status.get("value") if isinstance(status, dict) else status
                # same semantics as running_models(): loaded/sleeping keep KV
                return value in ("loaded", "sleeping")
            return False
        if self.direct_llama_server:
            return True  # single server: the forward never needs a load
        try:
            r = await self._client.get("/running")
            if r.status_code != 200:
                return None
            data = r.json()
        except (httpx.HTTPError, ValueError):
            return None
        names: list[str] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    names.append(item)
                elif isinstance(item, dict):
                    mid = item.get("id") or item.get("model") or item.get("name")
                    if mid:
                        names.append(str(mid))
        elif isinstance(data, dict):
            if "models" in data and isinstance(data["models"], list):
                names = [str(m) for m in data["models"]]
            else:
                names = list(data.keys())
        return model in names

    async def health(self, model: str | None = None) -> bool:
        """True when llama-swap (and the model's server, if given) is ready."""
        try:
            r = await self._client.get("/health")
            if r.status_code != 200:
                return False
            if model:
                # /upstream passthrough to the model's own health endpoint
                r2 = await self._client.get(f"/upstream/{model}/health")
                return r2.status_code == 200
            return True
        except httpx.HTTPError:
            return False

    async def get_models(self) -> Any:
        r = await self._client.get("/v1/models")
        return r.json() if r.status_code == 200 else {"data": []}

    # ---------------------------------------------------------- forwarding

    async def forward_json(
        self, path: str, body: dict, headers: dict[str, str] | None = None
    ) -> tuple[int, dict]:
        """Non-streaming forward. Returns (status_code, parsed_json)."""
        h = {"Content-Type": "application/json"}
        if headers:
            h.update(headers)
        r = await self._client.post(path, json=body, headers=h)
        try:
            data = r.json()
        except ValueError:
            data = {"error": {"message": r.text[:500]}}
        return r.status_code, data

    async def forward_stream(
        self, path: str, body: dict, headers: dict[str, str] | None = None
    ) -> AsyncIterator[tuple[int, dict, bytes]]:
        """Streaming forward. Yields (status, response_headers, chunk) tuples.

        The first yielded item carries the status line and headers; subsequent
        items carry raw body chunks. Caller is responsible for closing the
        response context (use as an async context manager around aiter_bytes).
        """
        h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if headers:
            h.update(headers)
        req = self._client.build_request("POST", path, json=body, headers=h)
        resp = await self._client.send(req, stream=True)
        try:
            yield resp.status_code, dict(resp.headers), b""
            async for chunk in resp.aiter_bytes():
                if chunk:
                    yield resp.status_code, {}, chunk
        finally:
            await resp.aclose()
