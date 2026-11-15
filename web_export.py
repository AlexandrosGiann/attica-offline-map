"""attica.amap (xz) -> web/attica.bin (+ .gz) για τον browser.
Ο browser δεν έχει αποσυμπιεστή xz, οπότε ο server στέλνει gzip με Content-Encoding και
ο browser το αποσυμπιέζει μόνος του (δουλεύει και σε παλιούς browsers).
Χρήση: python web_export.py [attica.amap] [web]
"""
import gzip
import lzma
import os
import sys

import amap


def main(src="attica.amap", outdir="web"):
    with open(src, "rb") as f:
        raw = f.read()
    if not raw.startswith(amap.MAGIC):
        sys.exit("δεν είναι .amap")
    payload = lzma.decompress(raw[len(amap.MAGIC):])
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "attica.bin")
    for path, data in ((p, payload), (p + ".gz", gzip.compress(payload, 9, mtime=0))):
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    print(f"attica.bin {len(payload) / 1e6:.2f} MB, attica.bin.gz {os.path.getsize(p + '.gz') / 1e6:.2f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:3])
