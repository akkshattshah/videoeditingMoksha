"""Expose local files over public HTTPS via a cloudflared quick tunnel.

OpenRouter's video API rejects data URIs and only accepts HTTPS URLs, so a
local reference video has to be reachable from the internet. A cloudflared
quick tunnel does that for free with no account and no storage bucket.

The tunnel is anonymous and ephemeral - it dies with the process.
"""

from __future__ import annotations

import functools
import http.server
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

QUICK_TUNNEL_RE = re.compile(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com")
TUNNEL_TIMEOUT = 45.0
REACHABLE_TIMEOUT = 75.0


class TunnelError(RuntimeError):
    pass


def cloudflared_path() -> str:
    exe = shutil.which("cloudflared")
    if not exe:
        raise TunnelError(
            "cloudflared not found on PATH. Install it, or host the files "
            "somewhere with a public HTTPS URL."
        )
    return exe


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:  # keep the console clean
        pass


@contextmanager
def serve_public(directory: str | Path, verbose: bool = True):
    """Serve `directory` and yield its public HTTPS base URL.

    Usage:
        with serve_public("in") as base:
            url = f"{base}/reference.mp4"
    """
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise TunnelError(f"not a directory: {directory}")

    exe = cloudflared_path()
    port = _free_port()

    handler = functools.partial(_QuietHandler, directory=str(directory))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()

    # cloudflared auto-loads ~/.cloudflared/config.yml even when --url is
    # given. If that config defines a named tunnel with ingress rules, the
    # quick tunnel's *.trycloudflare.com hostname matches none of them and
    # falls through to the catch-all - typically `http_status:404`, which
    # looks exactly like a broken origin. Pointing at an empty config keeps
    # the quick tunnel isolated from any existing setup.
    empty_cfg = Path(tempfile.gettempdir()) / "vre_cloudflared_empty.yml"
    if not empty_cfg.exists():
        empty_cfg.write_text("\n", encoding="utf-8")

    proc = subprocess.Popen(
        [exe, "tunnel", "--config", str(empty_cfg),
         "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )

    public: str | None = None
    lines: list[str] = []
    deadline = time.time() + TUNNEL_TIMEOUT
    try:
        while time.time() < deadline and proc.poll() is None:
            line = proc.stdout.readline() if proc.stdout else ""
            if not line:
                time.sleep(0.05)
                continue
            lines.append(line)
            m = QUICK_TUNNEL_RE.search(line)
            if m:
                public = m.group(0)
                break

        if not public:
            tail = "".join(lines[-12:])
            raise TunnelError(f"no tunnel URL after {TUNNEL_TIMEOUT:.0f}s.\n{tail}")

        # Drain remaining output so the pipe never fills and blocks cloudflared.
        threading.Thread(
            target=lambda: [None for _ in iter(proc.stdout.readline, "")],
            daemon=True,
        ).start()

        if verbose:
            print(f"    tunnel  {public}")

        _wait_reachable(directory, public, verbose=verbose)
        yield public

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        httpd.shutdown()
        httpd.server_close()


def _wait_reachable(directory: Path, base: str, verbose: bool = True) -> None:
    """Block until the tunnel actually serves a file.

    Quick tunnels take several seconds to propagate. Handing OpenRouter a URL
    that 404s wastes a paid request, so this is verified before any spend.
    """
    probe = next(
        (p for p in sorted(directory.iterdir()) if p.is_file()), None
    )
    if probe is None:
        raise TunnelError(f"nothing to serve in {directory}")

    url = f"{base}/{probe.name}"
    deadline = time.time() + REACHABLE_TIMEOUT
    last = ""
    with httpx.Client(timeout=20.0, follow_redirects=True) as c:
        while time.time() < deadline:
            try:
                r = c.head(url)
                if r.status_code == 200:
                    if verbose:
                        print(f"    reachable ({probe.name})")
                    return
                last = f"HTTP {r.status_code}"
            except Exception as exc:
                last = f"{type(exc).__name__}"
            time.sleep(2.0)
    raise TunnelError(f"tunnel never became reachable ({last}): {url}")


def url_for(base: str, path: str | Path) -> str:
    from urllib.parse import quote

    return f"{base}/{quote(Path(path).name)}"
