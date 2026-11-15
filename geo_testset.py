"""Δοκιμαστικές φωτογραφίες που ΔΕΝ είναι στο πακέτο: για τυχαία αντικείμενα Wikidata
παίρνει άλλη φωτογραφία από την κατηγορία Commons τους (P373). -> build/wd/test/*.jpg + test.json
Χρήση: python geo_testset.py [πλήθος=30]
"""
import json
import os
import random
import sys
import time
import urllib.parse

from wd_fetch import get, OUT

API_WD = "https://www.wikidata.org/w/api.php"
API_C = "https://commons.wikimedia.org/w/api.php"


def main(n=30):
    items = json.load(open(os.path.join(OUT, "manifest.json"), encoding="utf-8"))
    used = {i["file"] for i in items}
    byq = {}
    for i in items:
        byq.setdefault(i["q"], i)
    qs = list(byq)
    random.Random(7).shuffle(qs)
    tdir = os.path.join(OUT, "test")
    os.makedirs(tdir, exist_ok=True)
    tests = []
    for k in range(0, len(qs), 50):
        if len(tests) >= n:
            break
        part = qs[k:k + 50]
        j = json.loads(get(API_WD, data=urllib.parse.urlencode(
            {"action": "wbgetentities", "ids": "|".join(part), "props": "claims", "format": "json"}).encode()))
        for q in part:
            if len(tests) >= n:
                break
            cl = j["entities"].get(q, {}).get("claims", {}).get("P373")
            if not cl:
                continue
            cat = cl[0]["mainsnak"].get("datavalue", {}).get("value")
            if not cat:
                continue
            r = json.loads(get(API_C, data=urllib.parse.urlencode(
                {"action": "query", "format": "json", "generator": "categorymembers", "gcmtitle": "Category:" + cat,
                 "gcmtype": "file", "gcmlimit": 30, "prop": "imageinfo", "iiprop": "url|mime",
                 "iiurlwidth": 500}).encode()))
            time.sleep(0.5)
            cands = [p for p in r.get("query", {}).get("pages", {}).values()
                     if p.get("imageinfo") and p["imageinfo"][0].get("mime") == "image/jpeg"
                     and p["title"][5:] not in used]
            if not cands:
                continue
            p = random.Random(q).choice(cands)
            b = get(p["imageinfo"][0]["thumburl"])
            if not b:
                continue
            fn = os.path.join(tdir, f"{q}.jpg")
            with open(fn, "wb") as f:
                f.write(b)
            it = byq[q]
            tests.append({"q": q, "lat": it["lat"], "lon": it["lon"], "name": it["el"] or it["en"],
                          "file": p["title"][5:], "path": fn})
            print(len(tests), q, it["el"] or it["en"], "<-", p["title"][5:], flush=True)
            time.sleep(1)
    json.dump(tests, open(os.path.join(OUT, "test.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
