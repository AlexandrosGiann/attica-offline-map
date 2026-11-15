"""Τοπικός server για τη web εκδοχή (μόνο για το τοπικό δίκτυο).
Χρήση: python server.py [port]   ->  http://<IP-του-υπολογιστή>:<port>/
Το attica.bin στέλνεται ως gzip (Content-Encoding) όταν ο browser το δέχεται.
"""
import http.server
import json
import os
import re
import socket
import subprocess
import sys
import threading
import uuid

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "web")
GEO_EXE = os.path.join(BASE, "geo", "target", "release", "attica-geo.exe")
GEO_DIR = os.path.join(BASE, "build", "geo_q")
GEO_LOCK = threading.Lock()          # ένα ερώτημα τη φορά (τα μοντέλα χρησιμοποιούν όλους τους πυρήνες)
EXT = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/bmp": "bmp", "image/gif": "gif"}


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def _json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        """POST /geo/query με τα bytes της φωτογραφίας -> top-5 σημεία (attica-geo)."""
        if self.path.split("?", 1)[0] != "/geo/query":
            return self._json(404, {"error": "άγνωστη διεύθυνση"})
        n = int(self.headers.get("Content-Length", 0) or 0)
        if not 0 < n <= 20 << 20:
            return self._json(413, {"error": "η φωτογραφία πρέπει να είναι έως 20 MB"})
        body = self.rfile.read(n)
        ext = EXT.get(self.headers.get("Content-Type", "").split(";")[0].strip(), "jpg")
        qid = uuid.uuid4().hex[:12]
        d = os.path.join(GEO_DIR, qid)
        os.makedirs(d, exist_ok=True)
        src = os.path.join(d, "upload." + ext)
        with open(src, "wb") as f:
            f.write(body)
        with GEO_LOCK:
            r = subprocess.run([GEO_EXE, "query", src, "--out", d], cwd=BASE, capture_output=True, timeout=180)
        if r.returncode != 0:
            return self._json(500, {"error": r.stderr.decode("utf-8", "replace")[-500:]})
        with open(os.path.join(d, "result.json"), encoding="utf-8") as f:
            res = json.load(f)
        res["id"] = qid
        res["base"] = "/geo/r/%s/" % qid
        return self._json(200, res)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        m = re.fullmatch(r"/geo/r/([0-9a-f]{12})/([A-Za-z0-9_.]+)", path)
        if m:
            fp = os.path.join(GEO_DIR, m.group(1), m.group(2))
            if not os.path.isfile(fp):
                return self._json(404, {"error": "δεν υπάρχει"})
            ct = {"webp": "image/webp", "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "json": "application/json",
                  "html": "text/html; charset=utf-8"}.get(fp.rsplit(".", 1)[-1].lower(), "application/octet-stream")
            with open(fp, "rb") as f:
                b = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        gz = os.path.join(ROOT, "attica.bin.gz")
        if path == "/attica.bin" and "gzip" in self.headers.get("Accept-Encoding", "") and os.path.isfile(gz):
            size = os.path.getsize(gz)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with open(gz, "rb") as f:
                while True:
                    b = f.read(1 << 20)
                    if not b:
                        break
                    try:
                        self.wfile.write(b)
                    except (BrokenPipeError, ConnectionResetError):
                        return
            return
        return super().do_GET()

    def log_message(self, fmt, *args):
        sys.stdout.write("%s %s\n" % (self.address_string(), fmt % args))
        sys.stdout.flush()


def lan_ips():
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except OSError:
        pass
    return sorted(ips)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    if not os.path.isfile(os.path.join(ROOT, "index.html")):
        sys.exit(f"λείπει {ROOT}\\index.html")
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", port), H)
    print(f"Σερβίρω {ROOT}")
    print(f"  τοπικά:  http://localhost:{port}/")
    for ip in lan_ips():
        print(f"  δίκτυο:  http://{ip}:{port}/   φωτογραφία -> θέση: http://{ip}:{port}/geo.html")
    sys.stdout.flush()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
