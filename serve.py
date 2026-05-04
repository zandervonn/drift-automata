#!/usr/bin/env python3
"""Tiny static server for the SmoothLife app.

Run from the IDE (right-click -> Run) or:
    python serve.py
then open http://localhost:8000/ in a browser.
"""
import http.server
import os
import socketserver
import sys
import webbrowser

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
ROOT = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js":  "application/javascript; charset=utf-8",
        ".mjs": "application/javascript; charset=utf-8",
        ".html": "text/html; charset=utf-8",
    }

    def end_headers(self):
        # Don't cache during dev — edits should be visible on reload.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


def main():
    os.chdir(ROOT)
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/"
        print(f"smoothlife serving at {url}  (Ctrl+C to stop)")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
