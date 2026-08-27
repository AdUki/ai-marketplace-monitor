"""FastAPI app factory and uvicorn-in-a-thread runner.

The monitor process stays fully synchronous. Uvicorn runs on its own
asyncio loop in a daemon thread; the LogBroadcastHandler bridges records
from the main thread to that loop via ``loop.call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import secrets
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import uvicorn
from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    Form,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..utils import cache
from .auth import (
    CSRF_COOKIE,
    CSRF_HEADER,
    SESSION_COOKIE,
    SESSION_TTL,
    AuthConfig,
    RateLimiter,
    SessionManager,
    hash_password,
    verify_password,
)
from .config_api import ConfigFileService
from .config_auth import extract_credentials
from .found_export import build_found_rows, iter_found_csv, iter_found_rows
from .log_handler import LogBroadcastHandler

# Ensure the vendored toml-edit-js WASM bundle is served with the right
# Content-Type. Python's mimetypes module learned .wasm in 3.10 but
# explicit registration is safer across patch versions.
mimetypes.add_type("application/wasm", ".wasm")

STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class WebUIConfig:
    host: str = "127.0.0.1"
    port: int = 8467
    config_files: List[Path] = field(default_factory=list)
    log_handler: LogBroadcastHandler | None = None


@dataclass
class StartupInfo:
    """Information about the running server, shown in the startup banner."""

    urls: List[str]
    username: str | None  # None in open mode
    host: str
    port: int
    exposed: bool


class AuthState:
    """Mutable auth state.

    On loopback (default) the web UI is always open — no password
    required.  When ``--webui-host`` exposes the server on a
    non-loopback interface, ``auth`` must be set (credentials from
    a marketplace config section or environment variables).
    """

    def __init__(self) -> None:
        self.auth: AuthConfig | None = None
        self.exposed: bool = False


def _resolve_auth(config: WebUIConfig) -> tuple[AuthState, StartupInfo]:
    """Build initial AuthState from config files and environment.

    On loopback the UI is always open.  When exposed (--webui-host),
    credentials are required — checked from ``[marketplace.*]`` config
    sections, then ``FACEBOOK_USERNAME`` / ``FACEBOOK_PASSWORD`` env
    vars.
    """
    exposed = config.host not in ("127.0.0.1", "localhost", "::1")
    state = AuthState()
    state.exposed = exposed

    if exposed:
        extracted = extract_credentials(config.config_files)
        if extracted.username and extracted.password:
            state.auth = AuthConfig(
                username=extracted.username,
                password_hash=hash_password(extracted.password),
                secret_key=secrets.token_urlsafe(32),
            )
        # If exposed with no credentials, start_webui() will reject this.

    info = StartupInfo(
        urls=_enumerate_urls(config.host, config.port),
        username=state.auth.username if state.auth else None,
        host=config.host,
        port=config.port,
        exposed=exposed,
    )
    return state, info


def _set_session_cookies(response: Response, token: str, csrf: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=SESSION_TTL,
        httponly=True,
        samesite="strict",
    )
    response.set_cookie(
        CSRF_COOKIE,
        csrf,
        max_age=SESSION_TTL,
        httponly=False,  # JS reads this to echo via header
        samesite="strict",
    )


def _enumerate_urls(host: str, port: int) -> List[str]:
    if host in ("127.0.0.1", "localhost", "::1"):
        return [f"http://127.0.0.1:{port}"]
    if host in ("0.0.0.0", "::"):  # noqa: S104 — intentional bind-all
        # Enumerate local interface addresses so the user sees every reachable URL.
        urls = [f"http://127.0.0.1:{port}"]
        try:
            hostname = socket.gethostname()
            for info in socket.getaddrinfo(hostname, None):
                addr = str(info[4][0])
                if addr and addr not in ("127.0.0.1", "::1"):
                    if ":" in addr:
                        urls.append(f"http://[{addr}]:{port}")
                    else:
                        urls.append(f"http://{addr}:{port}")
        except socket.gaierror:
            pass
        # De-duplicate preserving order.
        seen: set[str] = set()
        unique: List[str] = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique.append(url)
        return unique
    return [f"http://{host}:{port}"]


def create_app(
    config: WebUIConfig,
    state: AuthState,
    config_service: ConfigFileService,
    log_handler: LogBroadcastHandler,
) -> FastAPI:
    app = FastAPI(
        title="AI Marketplace Monitor",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    process_secret = secrets.token_urlsafe(32)
    sessions = SessionManager(process_secret)
    rate_limiter = RateLimiter()

    def is_open() -> bool:
        """True when running on loopback — no password required."""
        return not state.exposed

    def require_session(
        request: Request,
        session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
    ) -> str:
        if is_open():
            return "anonymous"
        if session is None:
            raise HTTPException(status_code=401, detail="Not authenticated")
        username = sessions.validate(session)
        if username is None:
            raise HTTPException(status_code=401, detail="Session expired")
        return username

    def require_csrf(
        request: Request,
        csrf_cookie: str | None = Cookie(default=None, alias=CSRF_COOKIE),
    ) -> None:
        if is_open():
            return  # open mode skips CSRF (nothing to protect)
        header = request.headers.get(CSRF_HEADER)
        if not header or not csrf_cookie or not secrets.compare_digest(header, csrf_cookie):
            raise HTTPException(status_code=403, detail="CSRF token mismatch")

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/api/auth/info")
    async def auth_info() -> Dict[str, Any]:
        """Return auth mode info for the frontend login screen."""
        return {
            "open": is_open(),
            "username_hint": state.auth.username if state.auth else None,
        }

    @app.post("/api/login")
    async def login(
        request: Request,
        response: Response,
        username: str = Form(""),
        password: str = Form(""),
    ) -> Dict[str, Any]:
        # Loopback — always open, no password needed.
        if is_open():
            token, csrf = sessions.issue("anonymous")
            _set_session_cookies(response, token, csrf)
            return {"username": "anonymous", "csrf": csrf}

        # Exposed — credentials required.
        client_ip = request.client.host if request.client else "unknown"
        if rate_limiter.is_locked(client_ip):
            raise HTTPException(status_code=429, detail="Too many failed attempts")

        assert state.auth is not None  # enforced by start_webui()
        if username != state.auth.username or not verify_password(
            password, state.auth.password_hash
        ):
            rate_limiter.record_failure(client_ip)
            raise HTTPException(status_code=401, detail="Invalid credentials")

        rate_limiter.reset(client_ip)
        token, csrf = sessions.issue(username)
        _set_session_cookies(response, token, csrf)
        return {"username": username, "csrf": csrf}

    @app.post("/api/logout")
    async def logout(response: Response) -> Dict[str, Any]:
        response.delete_cookie(SESSION_COOKIE)
        response.delete_cookie(CSRF_COOKIE)
        return {"ok": True}

    @app.get("/api/status")
    async def status(_: str = Depends(require_session)) -> Dict[str, Any]:
        files = config_service.list_files()
        return {
            "config_files": [f.__dict__ for f in files],
            "urls": _enumerate_urls(config.host, config.port),
            "auth_mode": "open" if is_open() else "authenticated",
            "open": is_open(),
            "vnc_enabled": os.environ.get("AIMM_ENABLE_VNC") == "1"
            and Path(os.environ.get("AIMM_NOVNC_DIR", "/usr/share/novnc")).is_dir(),
        }

    @app.get("/api/config/files")
    async def list_config_files(_: str = Depends(require_session)) -> Dict[str, Any]:
        return {"files": [f.__dict__ for f in config_service.list_files()]}

    @app.get("/api/config/file/{file_id}")
    async def get_config_file(file_id: str, _: str = Depends(require_session)) -> Dict[str, Any]:
        try:
            content, mtime = config_service.read(file_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from None
        from .config_api import scan_sections
        from .secrets_redact import MASK, has_mask

        sections = [
            {
                "name": s.name,
                "prefix": s.prefix,
                "suffix": s.suffix,
                "line_start": s.line_start,
                "line_end": s.line_end,
                "fields": s.fields,
            }
            for s in scan_sections(content)
        ]
        return {
            "content": content,
            "mtime": mtime,
            "has_masked_secrets": has_mask(content),
            "mask_token": MASK,
            "sections": sections,
        }

    @app.put("/api/config/file/{file_id}", response_model=None)
    async def put_config_file(
        file_id: str,
        body: Dict[str, Any],
        _: str = Depends(require_session),
        __: None = Depends(require_csrf),
    ) -> Dict[str, Any]:
        content = body.get("content")
        if not isinstance(content, str):
            raise HTTPException(status_code=400, detail="Missing 'content' field")
        base_mtime = body.get("base_mtime")
        try:
            new_mtime, ok, error = config_service.write(
                file_id, content, base_mtime if isinstance(base_mtime, (int, float)) else None
            )
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from None
        if not ok:
            status_code = 409 if error and "conflict" in error else 400
            return JSONResponse(  # type: ignore[return-value]
                status_code=status_code,
                content={"ok": False, "error": error, "mtime": new_mtime},
            )
        return {"ok": True, "mtime": new_mtime}

    @app.post("/api/config/validate")
    async def validate_config(
        body: Dict[str, Any],
        _: str = Depends(require_session),
        __: None = Depends(require_csrf),
    ) -> Dict[str, Any]:
        content = body.get("content")
        if not isinstance(content, str):
            raise HTTPException(status_code=400, detail="Missing 'content' field")
        ok, error = config_service.validate(content)
        return {"valid": ok, "error": error}

    @app.post("/api/monitor/restart")
    async def restart_monitor(
        _: str = Depends(require_session),
        __: None = Depends(require_csrf),
    ) -> Dict[str, Any]:
        """Wake the monitor by touching the config file.

        The file watcher interrupts the monitor's doze() sleep, causing
        it to reload the config and run all scheduled searches immediately.
        """
        try:
            path = config_service.editable_path
            path.touch()
            return {"ok": True, "message": "Monitor woken — searching all items now."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to touch config: {e}") from e

    @app.get("/api/logs")
    async def get_logs(
        limit: int = 500,
        level: str = "DEBUG",
        kind: str | None = None,
        item: str | None = None,
        min_score: int | None = None,
        _: str = Depends(require_session),
    ) -> Dict[str, Any]:
        level_value = logging.getLevelName(level.upper())
        if not isinstance(level_value, int):
            level_value = 0
        return {
            "records": log_handler.snapshot(
                limit=limit,
                min_level=level_value,
                kind=kind,
                item=item,
                min_score=min_score,
            ),
            "capacity": log_handler._buffer.maxlen,
        }

    @app.websocket("/ws/stream")
    async def ws_stream(websocket: WebSocket) -> None:
        # In open mode (loopback) skip cookie check; otherwise require
        # a valid session cookie on the WebSocket handshake.
        if not is_open():
            session = websocket.cookies.get(SESSION_COOKIE)
            if not session or sessions.validate(session) is None:
                await websocket.close(code=4401)
                return

        await websocket.accept()
        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1000)
        log_handler.subscribe(queue)
        try:
            # Send a brief hello so clients know the stream is live.
            await websocket.send_json({"type": "hello", "time": time.time()})
            while True:
                payload = await queue.get()
                await websocket.send_json({"type": "log", "record": payload})
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: S110 — client disconnected; nothing to handle
            pass
        finally:
            log_handler.unsubscribe(queue)

    # ------------------------------------------------------------------
    # Optional noVNC bridge (Docker deployments)
    # ------------------------------------------------------------------
    novnc_dir = os.environ.get("AIMM_NOVNC_DIR", "/usr/share/novnc")
    vnc_host = os.environ.get("AIMM_VNC_HOST", "127.0.0.1")
    vnc_port = int(os.environ.get("AIMM_VNC_PORT", "5900"))
    if os.environ.get("AIMM_ENABLE_VNC") == "1" and Path(novnc_dir).is_dir():
        app.mount("/vnc", StaticFiles(directory=novnc_dir, html=True), name="vnc")

        @app.websocket("/ws/vnc")
        async def ws_vnc(websocket: WebSocket) -> None:
            if not is_open():
                session = websocket.cookies.get(SESSION_COOKIE)
                if not session or sessions.validate(session) is None:
                    await websocket.close(code=4401)
                    return
            await websocket.accept(subprotocol="binary")
            try:
                reader, writer = await asyncio.open_connection(vnc_host, vnc_port)
            except OSError:
                await websocket.close(code=1011)
                return

            async def ws_to_tcp() -> None:
                try:
                    while True:
                        data = await websocket.receive_bytes()
                        writer.write(data)
                        await writer.drain()
                except WebSocketDisconnect:
                    pass
                finally:
                    writer.close()

            async def tcp_to_ws() -> None:
                try:
                    while True:
                        chunk = await reader.read(65536)
                        if not chunk:
                            break
                        await websocket.send_bytes(chunk)
                finally:
                    try:
                        await websocket.close()
                    except Exception:  # noqa: S110 — already closed
                        pass

            await asyncio.gather(ws_to_tcp(), tcp_to_ws(), return_exceptions=True)

    # ------------------------------------------------------------------
    # Static UI
    # ------------------------------------------------------------------
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    # Sync def (not async): FastAPI runs it in a threadpool and Starlette
    # iterates the sync generator there too, so the blocking cache scan never
    # runs on the event loop. The body streams row-by-row rather than buffering
    # the whole CSV, keeping memory bounded for large exports.
    @app.get("/api/found.csv")
    def export_found_csv(_: str = Depends(require_session)) -> StreamingResponse:
        filename = f"found-items-{time.strftime('%Y%m%d-%H%M%S')}.csv"
        return StreamingResponse(
            iter_found_csv(iter_found_rows(cache)),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/items")
    def api_items(_: str = Depends(require_session)) -> JSONResponse:
        """Everything known about each matched listing, for the item browser.

        Joins the notified rows with cached listing details and AI ratings
        (build_found_rows), then enriches each with the verified device facts
        looked up per model. Read-only and cache-only: no scraping, no network.
        """
        rows = build_found_rows(cache)
        try:
            from ..device_facts import facts_for_listing
        except Exception:  # noqa: BLE001
            facts_for_listing = None  # type: ignore[assignment]

        out = []
        for r in rows:
            item = dict(r)
            if facts_for_listing is not None:
                try:
                    f = facts_for_listing(r.get("title", "")) or {}
                    item["specs"] = {
                        k: f.get(k)
                        for k in (
                            "chip", "benchmark_name", "benchmark_score", "ram_gb",
                            "storage_gb", "used_price_eur", "release_year", "score",
                            "verdict", "confidence", "lineageos_official",
                            "lineageos_unofficial",
                        )
                        if f.get(k) is not None
                    }
                except Exception:  # noqa: BLE001 - browser must still render
                    item["specs"] = {}
            out.append(item)
        out.reverse()  # newest first
        return JSONResponse({"items": out, "count": len(out)})

    @app.get("/items")
    def items_page(_: str = Depends(require_session)) -> HTMLResponse:
        return HTMLResponse(ITEMS_PAGE)

    return app


# Mobile-first item browser. Deliberately self-contained (no build step, no
# framework, no external requests) and served as one string: the existing SPA
# is a 67KB bundle whose layout assumes a desktop viewport, and this needs to
# be usable one-handed on a phone while standing in front of a seller.
ITEMS_PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Deals</title>
<style>
:root{--bg:#0f1115;--card:#181b22;--line:#272b35;--fg:#e8eaed;--dim:#9aa0aa;
      --good:#37b24d;--ok:#f59f00;--bad:#e03131;--link:#4dabf7}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
     padding:env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left)}
header{position:sticky;top:0;z-index:5;background:var(--bg);
       border-bottom:1px solid var(--line);padding:10px 12px}
h1{font-size:17px;margin:0 0 8px}
.controls{display:flex;gap:8px;flex-wrap:wrap}
input,select{background:var(--card);color:var(--fg);border:1px solid var(--line);
             border-radius:8px;padding:9px 10px;font-size:16px;flex:1 1 130px;min-width:0}
main{padding:12px;display:grid;gap:12px;max-width:900px;margin:0 auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;overflow-wrap:anywhere}
.top{display:flex;gap:10px;align-items:flex-start;justify-content:space-between}
.title{font-weight:600;margin:0;font-size:15px}
.badge{flex:none;border-radius:999px;padding:3px 9px;font-size:12px;font-weight:700;color:#0b0d10}
.r5,.r4{background:var(--good)}.r3{background:var(--ok)}.r2,.r1{background:var(--bad)}
.price{font-size:20px;font-weight:700;margin:6px 0 2px}
.meta{color:var(--dim);font-size:13px}
.specs{margin-top:9px;display:flex;flex-wrap:wrap;gap:6px}
.chip{background:#20242c;border:1px solid var(--line);border-radius:7px;
      padding:3px 7px;font-size:12px;color:var(--dim)}
.chip b{color:var(--fg);font-weight:600}
.q{font-weight:700}.q.good{color:var(--good)}.q.ok{color:var(--ok)}.q.weak{color:var(--bad)}
.ai{margin-top:9px;font-size:13px;color:var(--fg);background:#12151b;
    border-left:3px solid var(--link);padding:7px 9px;border-radius:0 7px 7px 0}
.desc{margin-top:8px;font-size:13px;color:var(--dim);white-space:pre-wrap;
      max-height:4.4em;overflow:hidden}
.desc.open{max-height:none}
a.go{display:inline-block;margin-top:10px;background:var(--link);color:#08101c;
     text-decoration:none;font-weight:700;padding:9px 13px;border-radius:9px}
.more{margin-top:6px;color:var(--link);font-size:12px;cursor:pointer;
      background:none;border:0;padding:0}
.empty{color:var(--dim);text-align:center;padding:36px 12px}
</style></head><body>
<header>
  <h1>Deals <span id="n" class="meta"></span></h1>
  <div class="controls">
    <input id="q" placeholder="Search title, seller, AI note...">
    <select id="min"><option value="0">any rating</option><option value="3">3+</option>
      <option value="4">4+</option><option value="5">5 only</option></select>
    <select id="sort"><option value="new">newest</option><option value="rating">rating</option>
      <option value="quality">quality</option><option value="cheap">cheapest</option></select>
  </div>
</header>
<main id="list"><div class="empty">Loading...</div></main>
<script>
const E=(s)=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const num=(s)=>{const m=String(s??"").replace(/[ ,]/g,"").match(/-?\\d+(\\.\\d+)?/);return m?parseFloat(m[0]):null};
let ITEMS=[];
function specChips(s){
  if(!s) return "";
  const c=[];
  if(s.chip) c.push(`<span class="chip"><b>${E(s.chip)}</b></span>`);
  if(s.benchmark_score) c.push(`<span class="chip">${E((s.benchmark_name||"bench").replace("_"," "))} <b>${s.benchmark_score.toLocaleString()}</b></span>`);
  if(s.ram_gb&&s.ram_gb.length) c.push(`<span class="chip">RAM <b>${s.ram_gb.join("/")}GB</b></span>`);
  if(s.storage_gb&&s.storage_gb.length) c.push(`<span class="chip">disk <b>${s.storage_gb.join("/")}GB</b></span>`);
  if(s.used_price_eur) c.push(`<span class="chip">used ~<b>EUR ${s.used_price_eur}</b></span>`);
  if(s.release_year) c.push(`<span class="chip">${s.release_year}</span>`);
  if(s.lineageos_official) c.push(`<span class="chip"><b>LineageOS</b> official</span>`);
  else if(s.lineageos_unofficial) c.push(`<span class="chip"><b>LineageOS</b> unofficial</span>`);
  if(s.score!=null){
    const k=s.score>=55?"good":s.score>=35?"ok":"weak";
    c.push(`<span class="chip">quality <b class="q ${k}">${s.score}/100</b>${s.confidence==="low"?" (unverified)":""}</span>`);
  }
  return `<div class="specs">${c.join("")}</div>`;
}
function card(it,i){
  const r=parseInt(it.rating)||0, s=it.specs||{};
  return `<div class="card">
    <div class="top"><p class="title">${E(it.title)}</p>
      ${r?`<span class="badge r${r}">${r}</span>`:""}</div>
    <div class="price">${E(it.price||"?")}</div>
    <div class="meta">${E(it.location||"")}${it.seller?" &middot; "+E(it.seller):""}${it.condition?" &middot; "+E(it.condition):""}</div>
    <div class="meta">${E(it.item||"")}${it.found_at?" &middot; "+E(it.found_at):""}</div>
    ${specChips(s)}
    ${it.ai_comment?`<div class="ai">${E(it.ai_comment)}</div>`:""}
    ${it.description?`<div class="desc" id="d${i}">${E(it.description)}</div>
      <button class="more" onclick="document.getElementById('d${i}').classList.toggle('open')">show more / less</button>`:""}
    ${it.url?`<a class="go" href="${E(it.url)}" target="_blank" rel="noopener">Open on Facebook</a>`:""}
  </div>`;
}
function render(){
  const q=document.getElementById("q").value.toLowerCase();
  const min=parseInt(document.getElementById("min").value)||0;
  const sort=document.getElementById("sort").value;
  let v=ITEMS.filter(it=>(parseInt(it.rating)||0)>=min).filter(it=>!q||
    [it.title,it.seller,it.ai_comment,it.item,it.description].join(" ").toLowerCase().includes(q));
  if(sort==="rating") v.sort((a,b)=>(parseInt(b.rating)||0)-(parseInt(a.rating)||0));
  if(sort==="quality") v.sort((a,b)=>((b.specs||{}).score??-1)-((a.specs||{}).score??-1));
  if(sort==="cheap") v.sort((a,b)=>(num(a.price)??1e9)-(num(b.price)??1e9));
  document.getElementById("n").textContent=`(${v.length}/${ITEMS.length})`;
  document.getElementById("list").innerHTML=v.length?v.map(card).join(""):
    '<div class="empty">Nothing matches. Matches appear here once the monitor rates a listing highly enough to notify.</div>';
}
fetch("/api/items",{credentials:"same-origin"}).then(r=>{
  if(r.status===401){location.href="/";return null} return r.json()})
 .then(d=>{if(!d)return;ITEMS=d.items||[];render()})
 .catch(e=>{document.getElementById("list").innerHTML=
   '<div class="empty">Could not load items: '+E(e.message)+'</div>'});
["q","min","sort"].forEach(id=>document.getElementById(id)
  .addEventListener("input",render));
</script></body></html>"""


# ----------------------------------------------------------------------
# Thread runner
# ----------------------------------------------------------------------


class WebUIServer:
    """Runs uvicorn in a background thread."""

    def __init__(
        self,
        config: WebUIConfig,
        state: AuthState,
        config_service: ConfigFileService,
    ) -> None:
        if config.log_handler is None:
            raise ValueError("WebUIConfig.log_handler is required")
        self._config = config
        self._state = state
        self._config_service = config_service
        self._app = create_app(config, state, config_service, config.log_handler)
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()

    def start(self) -> None:
        uv_config = uvicorn.Config(
            self._app,
            host=self._config.host,
            port=self._config.port,
            log_level="warning",
            access_log=False,
            lifespan="off",
        )
        self._server = uvicorn.Server(uv_config)

        def runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            assert self._config.log_handler is not None
            self._config.log_handler.attach_loop(loop)
            self._ready.set()
            try:
                loop.run_until_complete(self._server.serve())  # type: ignore[union-attr]
            finally:
                loop.close()

        self._thread = threading.Thread(target=runner, name="aimm-webui", daemon=True)
        self._thread.start()
        # Give the loop a moment to bind so attach_loop completes before
        # any log records are emitted.
        self._ready.wait(timeout=5)

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True


def start_webui(
    config: WebUIConfig, logger: logging.Logger | None = None
) -> tuple[WebUIServer, StartupInfo]:
    """Resolve auth, build the service, and start the server thread."""
    if config.log_handler is None:
        raise ValueError("WebUIConfig.log_handler is required")
    state, info = _resolve_auth(config)

    # --webui-host requires credentials. Refuse to expose without auth.
    if state.exposed and state.auth is None:
        raise RuntimeError(
            f"--webui-host {config.host} requires authentication. "
            "Set username/password in a [marketplace.*] config section "
            "or set FACEBOOK_USERNAME and FACEBOOK_PASSWORD environment "
            "variables. Omit --webui-host to run on 127.0.0.1 without "
            "a password."
        )

    config_service = ConfigFileService(config.config_files, logger=logger)
    server = WebUIServer(config, state, config_service)
    server.start()
    return server, info
