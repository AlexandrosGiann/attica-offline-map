"""build/wd (manifest + raw + feat) -> attica_geo.pack
Μορφή: b"AGEO1\\n" + u32 μήκος header + header JSON {"sections": {όνομα: [offset, μήκος]}} + ενότητες:
  meta  xz JSON (αντικείμενα Wikidata, εικόνες, ετικέτες, OCR)
  emb   int8 [κομμάτια][512]  (κάθε γραμμή με δική της κλίμακα)
  scale f32  [κομμάτια]
  cptr  u32  [εικόνες+1]  κομμάτια ανά εικόνα
  iptr  u64  [εικόνες+1]  θέσεις στο img
  img   WebP εικόνες η μία μετά την άλλη (διαβάζεται μόνο όποια χρειάζεται)
Χρήση: python geo_pack.py [μέγιστη πλευρά=384] [ποιότητα WebP=60]
"""
import io
import json
import lzma
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageOps

WD = os.path.join("build", "wd")
FEAT = os.environ.get("GEO_FEAT", os.path.join(WD, "feat"))
MANIFEST = os.environ.get("GEO_MANIFEST", os.path.join(WD, "manifest.json"))


def main(side=384, quality=60, out=os.environ.get("GEO_OUT", "attica_geo.pack")):
    items_in = json.load(open(MANIFEST, encoding="utf-8"))
    idx = json.load(open(os.path.join(FEAT, "images.json"), encoding="utf-8"))
    hashes, cptr = idx["img"], np.array(idx["cptr"], np.uint32)
    hpos = {h: i for i, h in enumerate(hashes)}
    emb = np.fromfile(os.path.join(FEAT, "emb.f32"), np.float32).reshape(-1, 512)
    assert emb.shape[0] == cptr[-1]
    ocr = {}
    p = os.path.join(FEAT, "ocr.jsonl")
    if os.path.isfile(p):
        for line in open(p, encoding="utf-8"):
            j = json.loads(line)
            ocr[j["img"]] = " ".join(t for t, s in j["text"] if s >= 0.5)

    # ετικέτες σκηνής από το text bank του MobileCLIP (κεντρικό κομμάτι)
    lab = json.load(open(os.path.join("ext", "mc", "text-bank", "labels.json"), encoding="utf-8"))["labels"]
    tb = np.fromfile(os.path.join("ext", "mc", "text-bank", "embeddings.f32"), np.float32).reshape(len(lab), 512)
    tags = np.argsort(-(emb[cptr[:-1]] @ tb.T), axis=1)[:, :3]

    # αντικείμενα Wikidata (ένα ανά Q) και εικόνες
    items, qpos = [], {}
    img_items = [[] for _ in hashes]
    for it in items_in:
        if it["q"] not in qpos:
            qpos[it["q"]] = len(items)
            items.append({k: it[k] for k in ("q", "lon", "lat", "el", "en", "desc")})
        k, i = qpos[it["q"]], hpos[it["img"]]
        if k not in img_items[i]:
            img_items[i].append(k)
    files = {}
    for it in items_in:
        files.setdefault(it["img"], it["file"])

    # int8 με κλίμακα ανά γραμμή
    sc = np.abs(emb).max(axis=1) / 127.0
    q8 = np.round(emb / sc[:, None]).astype(np.int8)
    cos = ((q8 * sc[:, None]) * emb).sum(1) / np.linalg.norm(q8 * sc[:, None], axis=1)
    # WebP
    blobs, iptr, px_raw, src_bytes = [], [0], 0, 0
    images = []
    for i, h in enumerate(hashes):
        fp = os.path.join(WD, "raw", h + ".jpg")
        src_bytes += os.path.getsize(fp)
        im = ImageOps.exif_transpose(Image.open(fp)).convert("RGB")
        im.thumbnail((side, side), Image.LANCZOS)
        b = io.BytesIO()
        im.save(b, "WEBP", quality=quality, method=6)
        blobs.append(b.getvalue())
        iptr.append(iptr[-1] + len(blobs[-1]))
        px_raw += im.width * im.height * 3
        images.append({"items": img_items[i], "file": files[h], "w": im.width, "h": im.height,
                       "tags": [lab[t] for t in tags[i]], "ocr": ocr.get(h, "")})
    stats = {"images": len(hashes), "items": len(items), "crops": int(cptr[-1]), "side": side, "webp_q": quality,
             "pixels_decoded_MB": round(px_raw / 1e6, 2), "source_jpeg_MB": round(src_bytes / 1e6, 2),
             "int8_cos_min": round(float(cos.min()), 6)}
    meta = lzma.compress(json.dumps({"items": items, "images": images, "stats": stats}, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8"),
                         format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC32, preset=9 | lzma.PRESET_EXTREME)
    secs = [("meta", meta), ("emb", q8.tobytes()), ("scale", sc.astype("<f4").tobytes()),
            ("cptr", cptr.astype("<u4").tobytes()), ("iptr", np.array(iptr, "<u8").tobytes()),
            ("img", b"".join(blobs))]
    # header με σταθερό μήκος (offsets εξαρτώνται από αυτό)
    def header(base):
        o, d = base, {}
        for n, b in secs:
            d[n] = [o, len(b)]
            o += len(b)
        return json.dumps({"sections": d, "version": 1}).encode()
    h = header(0)
    h = header(6 + 4 + len(h) + 64).ljust(len(h) + 64)
    with open(out + ".tmp", "wb") as f:
        f.write(b"AGEO1\n" + struct.pack("<I", len(h)) + h)
        for _, b in secs:
            f.write(b)
    os.replace(out + ".tmp", out)
    tot = os.path.getsize(out)
    print(json.dumps(stats, ensure_ascii=False))
    print(f"{out}: {tot / 1e6:.2f} MB  (" + ", ".join(f"{n} {len(b) / 1e6:.2f}" for n, b in secs) + ")")
    print(f"αποσυμπιεσμένο: meta {len(lzma.decompress(meta)) / 1e6:.2f} MB + embeddings float32 "
          f"{emb.nbytes / 1e6:.2f} MB + pixels {px_raw / 1e6:.1f} MB")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 384, int(a[1]) if len(a) > 1 else 60)
