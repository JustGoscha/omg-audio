#!/usr/bin/env python3
"""Serve the repo root for the web demo.

Desktop:  python3 tools/serve.py  →  http://localhost:8000/web/
Android:  connect phone via USB, enable USB debugging, then:
            adb reverse tcp:8000 tcp:8000
          and open http://localhost:8000/web/ in Chrome on the phone.
          (localhost is a secure context, so AudioWorklet + sensors work
          without HTTPS certificates.)
"""
import http.server
import functools
import os

PORT = 8000
root = os.path.join(os.path.dirname(__file__), "..")

class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

handler = functools.partial(NoCacheHandler, directory=root)
handler.extensions_map = getattr(handler, "extensions_map", {})
http.server.SimpleHTTPRequestHandler.extensions_map[".wasm"] = "application/wasm"
http.server.SimpleHTTPRequestHandler.extensions_map[".js"] = "text/javascript"

print(f"serving {os.path.abspath(root)} on http://localhost:{PORT}/web/")
http.server.ThreadingHTTPServer(("", PORT), handler).serve_forever()
