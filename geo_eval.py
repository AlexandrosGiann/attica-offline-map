"""Αξιολόγηση σε φωτογραφίες εκτός πακέτου (build/wd/test.json).
Χρήση: python geo_eval.py [--no-mirage] [--no-verify]
"""
import json
import math
import os
import subprocess
import sys

EXE = os.path.join("geo", "target", "release", "attica-geo.exe")


def dist(a, b, c, d):
    p1, p2 = math.radians(a), math.radians(c)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(d - b) / 2) ** 2
    return 12742000 * math.asin(math.sqrt(h))


def main(flags):
    tests = json.load(open(os.path.join("build", "wd", "test.json"), encoding="utf-8"))
    tag = "_".join(f.strip("-") for f in flags) or "full"
    rows, hit1, hit5, near1, near5, d1 = [], 0, 0, 0, 0, []
    for t in tests:
        out = os.path.join("build", "eval", tag, t["q"])
        subprocess.run([EXE, "query", t["path"], "--out", out, *flags], check=True, capture_output=True)
        r = json.load(open(os.path.join(out, "result.json"), encoding="utf-8"))
        res = r["results"]
        qs = [[p["q"] for p in x["places"]] for x in res]
        ds = [dist(t["lat"], t["lon"], x["lat"], x["lon"]) for x in res]
        rank = next((k + 1 for k, q in enumerate(qs) if t["q"] in q), None)
        hit1 += rank == 1
        hit5 += rank is not None
        near1 += ds[0] < 1000
        near5 += min(ds) < 1000
        d1.append(ds[0])
        rows.append((t["name"], rank, round(ds[0]), r["people_cover"], r["timing"]))
        print(f"{t['name'][:40]:40} rank={rank} top1 {ds[0] / 1000:.2f} km  άνθρωποι {r['people_cover']:.2f}", flush=True)
    n = len(tests)
    d1.sort()
    print(f"\n[{tag}] n={n}  σωστό αντικείμενο top1 {hit1}/{n}, top5 {hit5}/{n};  <1 km top1 {near1}/{n}, top5 {near5}/{n};"
          f"  διάμεση απόσταση top1 {d1[n // 2] / 1000:.2f} km")


if __name__ == "__main__":
    main(sys.argv[1:])
