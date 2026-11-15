"""Wikidata εικόνες της Αττικής -> build/wd/raw/*.jpg + build/wd/manifest.json
1) SPARQL: αντικείμενα με P625 στο bbox και εικόνα (P18, P5775 εσωτερικό, P3451 νύχτα,
   P8592 εναέρια, P4291 πανόραμα, P8517 θέα). 2) φίλτρο με το πολύγωνο της Περιφέρειας
   από το attica.amap. 3) thumbnails 500px από το Commons (τυπικό μέγεθος, με cache).
Χρήση: python wd_fetch.py [threads]
"""
import csv
import hashlib
import io
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import amap

UA = "attica-map-builder/0.1 (offline research; python-urllib)"
OUT = os.path.join("build", "wd")
RAW = os.path.join(OUT, "raw")
BBOX = (22.85, 35.80, 24.20, 38.42)
PROPS = ["P18", "P5775", "P3451", "P8592", "P4291", "P8517"]
WIDTH = 500
Q = """SELECT ?item ?coord ?prop ?image ?label_el ?label_en ?desc_en WHERE {
  SERVICE wikibase:box { ?item wdt:P625 ?coord .
    bd:serviceParam wikibase:cornerSouthWest "Point(%f %f)"^^geo:wktLiteral .
    bd:serviceParam wikibase:cornerNorthEast "Point(%f %f)"^^geo:wktLiteral . }
  VALUES ?prop { %s }
  ?item ?prop ?image .
  OPTIONAL { ?item rdfs:label ?label_el FILTER(LANG(?label_el)="el") }
  OPTIONAL { ?item rdfs:label ?label_en FILTER(LANG(?label_en)="en") }
  OPTIONAL { ?item schema:description ?desc_en FILTER(LANG(?desc_en)="en") }
}"""


def get(url, tries=9, data=None):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code not in (429, 500, 502, 503, 504) or k == tries - 1:
                raise
            time.sleep(int(e.headers.get("Retry-After", 0) or 0) or 2 ** k * 2)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def thumb_urls(files):
    """Commons API, 50 τίτλοι ανά αίτημα: το έτοιμο thumbnail URL (τυπικό πλάτος)."""
    cache = os.path.join(OUT, "thumbs.json")
    out = {}
    if os.path.isfile(cache):
        with open(cache, encoding="utf-8") as f:
            out = json.load(f)
    todo = [f for f in files if f not in out]
    for k in range(0, len(todo), 50):
        part = todo[k:k + 50]
        qs = urllib.parse.urlencode({"action": "query", "format": "json", "prop": "imageinfo", "iiprop": "url",
                                     "iiurlwidth": WIDTH, "titles": "|".join("File:" + f for f in part)})
        j = json.loads(get("https://commons.wikimedia.org/w/api.php", data=qs.encode()))
        norm = {n["to"]: n["from"] for n in j["query"].get("normalized", [])}
        for p in j["query"]["pages"].values():
            ii = p.get("imageinfo")
            if ii:
                t = norm.get(p["title"], p["title"])
                out[t[5:]] = ii[0].get("thumburl") or ii[0]["url"]
        time.sleep(0.5)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return out


def sparql():
    cache = os.path.join(OUT, "items2.csv")       # το WDQS κόβει επαναλαμβανόμενα αιτήματα
    if os.path.isfile(cache):
        with open(cache, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    q =Q % (BBOX[0], BBOX[1], BBOX[2], BBOX[3], " ".join("wdt:" + p for p in PROPS))
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": q})
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/csv"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return list(csv.DictReader(io.StringIO(r.read().decode("utf-8"))))


def main(threads=4):
    os.makedirs(RAW, exist_ok=True)
    rows = sparql()
    d = amap.load("attica.amap")
    sc = 1e6
    bx, by = np.cumsum(d["bnd.dx"]) / sc, np.cumsum(d["bnd.dy"]) / sc
    o = np.concatenate([[0], np.cumsum(d["bnd.n"])])
    poly = amap.Polygon([np.column_stack([bx[o[i]:o[i + 1]], by[o[i]:o[i + 1]]]) for i in range(len(o) - 1)])
    items, seen = [], set()
    for r in rows:
        c = r["coord"]
        if not c.startswith("Point("):
            continue
        lon, lat = map(float, c[6:-1].split())
        fn = urllib.parse.unquote(r["image"].rsplit("/", 1)[1])
        ext = fn.rsplit(".", 1)[-1].lower()
        if ext in ("svg", "gif", "webm", "ogv", "pdf", "djvu"):
            continue                       # λογότυπα/χάρτες/βίντεο: όχι φωτογραφίες
        key = (r["item"], fn)
        if key in seen:
            continue
        seen.add(key)
        items.append(dict(q=r["item"].rsplit("/", 1)[1], lon=lon, lat=lat, prop=r["prop"].rsplit("/", 1)[1],
                          file=fn, el=r["label_el"], en=r["label_en"], desc=r["desc_en"]))
    ins = poly.contains(np.array([i["lon"] for i in items]), np.array([i["lat"] for i in items]))
    items = [i for i, k in zip(items, ins) if k]
    print(f"SPARQL γραμμές {len(rows)}, μέσα στην Αττική {len(items)}, αρχεία {len({i['file'] for i in items})}")

    lock = threading.Lock()
    stat = {"ok": 0, "cached": 0, "miss": 0, "bytes": 0}

    def fetch(fn):
        h = hashlib.sha1(fn.encode("utf-8")).hexdigest()[:16]
        p = os.path.join(RAW, h + ".jpg")
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            with lock:
                stat["cached"] += 1
            return fn, h
        url = thumbs.get(fn)
        if not url:
            with lock:
                stat["miss"] += 1
            return fn, None
        try:
            b = get(url)
        except Exception as e:  # noqa: BLE001
            print("σφάλμα", fn, e)
            b = None
        with lock:
            if not b:
                stat["miss"] += 1
                return fn, None
            stat["ok"] += 1
            stat["bytes"] += len(b)
            n = stat["ok"] + stat["cached"] + stat["miss"]
            if n % 100 == 0:
                print(f"  {n}  {stat}", flush=True)
        with open(p + ".tmp", "wb") as f:
            f.write(b)
        os.replace(p + ".tmp", p)
        return fn, h

    files = sorted({i["file"] for i in items})
    thumbs = thumb_urls(files)
    print(f"thumbnail URLs {len(thumbs)}/{len(files)}", flush=True)
    t = time.time()
    with ThreadPoolExecutor(threads) as ex:
        res = dict(ex.map(fetch, files))
    for i in items:
        i["img"] = res.get(i["file"])
    items = [i for i in items if i["img"]]
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    print(f"έτοιμο σε {time.time() - t:.0f} s: {stat}, εγγραφές {len(items)}")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:2]))
