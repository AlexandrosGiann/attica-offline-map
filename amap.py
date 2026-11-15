"""Μορφή αρχείου .amap: ένα xz/LZMA stream με ονομασμένες ενότητες.
Ακέραιοι πίνακες: zigzag + varint (numpy, διανυσματικά) -> πολύ μικροί μετά το LZMA.

header: b"AMAP1\\n" + xz( ενότητες )
ενότητα: u16 len(name) | name | u8 kind | u64 len(data) | data
kind: 0 bytes, 1 uvarint[], 2 svarint[], 3 json(utf-8), 4 strings ('\\0'-joined utf-8)
"""
import json
import lzma
import struct

import numpy as np

MAGIC = b"AMAP1\n"
SCALE = 1_000_000  # συντεταγμένες σε 1e-6 μοίρες (~0.11 m)


# ---------------- varint ----------------
def uvarint_encode(a):
    a = np.ascontiguousarray(a, dtype=np.uint64).ravel()
    if a.size == 0:
        return b""
    nb = np.ones(a.size, dtype=np.int64)
    for k in range(1, 10):
        nb += a >= np.uint64(1 << (7 * k))
    start = np.zeros(a.size, dtype=np.int64)
    np.cumsum(nb[:-1], out=start[1:])
    out = np.empty(int(nb.sum()), dtype=np.uint8)
    for k in range(int(nb.max())):
        m = nb > k
        v = (a[m] >> np.uint64(7 * k)) & np.uint64(0x7F)
        v = v.astype(np.uint8) | np.where(nb[m] > k + 1, 0x80, 0).astype(np.uint8)
        out[start[m] + k] = v
    return out.tobytes()


def uvarint_decode(b):
    b = np.frombuffer(b, dtype=np.uint8)
    if b.size == 0:
        return np.zeros(0, dtype=np.uint64)
    ends = np.flatnonzero(b < 0x80)
    starts = np.empty_like(ends)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    ln = ends - starts + 1
    out = np.zeros(ends.size, dtype=np.uint64)
    for k in range(int(ln.max())):
        m = ln > k
        out[m] |= (b[starts[m] + k] & 0x7F).astype(np.uint64) << np.uint64(7 * k)
    return out


def zz_enc(a):
    a = np.asarray(a, dtype=np.int64)
    return ((a << 1) ^ (a >> 63)).astype(np.uint64)


def zz_dec(u):
    u = np.asarray(u, dtype=np.uint64)
    return ((u >> np.uint64(1)).astype(np.int64)) ^ (-(u & np.uint64(1)).astype(np.int64))


# ---------------- container ----------------
class Writer:
    def __init__(self):
        self.parts = []

    def _add(self, name, kind, data):
        n = name.encode()
        self.parts.append(struct.pack("<H", len(n)) + n + struct.pack("<BQ", kind, len(data)) + data)

    def bytes_(self, name, b):
        self._add(name, 0, bytes(b))

    def uints(self, name, a):
        self._add(name, 1, uvarint_encode(a))

    def ints(self, name, a):
        self._add(name, 2, uvarint_encode(zz_enc(a)))

    def json(self, name, obj):
        self._add(name, 3, json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode())

    def strings(self, name, lst):
        for s in lst:
            if "\0" in s:
                raise ValueError("NUL σε string")
        self._add(name, 4, "\0".join(lst).encode())

    def save(self, path):
        payload = b"".join(self.parts)
        filt = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME,
                 "dict_size": 1 << 27}]
        data = MAGIC + lzma.compress(payload, format=lzma.FORMAT_XZ, filters=filt)
        tmp = str(path) + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        import os
        os.replace(tmp, path)
        return len(payload), len(data)


def load(path):
    with open(path, "rb") as f:
        raw = f.read()
    if not raw.startswith(MAGIC):
        raise ValueError("δεν είναι αρχείο .amap")
    p = lzma.decompress(raw[len(MAGIC):])
    out, i = {}, 0
    while i < len(p):
        (nl,) = struct.unpack_from("<H", p, i)
        i += 2
        name = p[i:i + nl].decode()
        i += nl
        kind, ln = struct.unpack_from("<BQ", p, i)
        i += 9
        d = p[i:i + ln]
        i += ln
        if kind == 0:
            out[name] = d
        elif kind == 1:
            out[name] = uvarint_decode(d).astype(np.int64)
        elif kind == 2:
            out[name] = zz_dec(uvarint_decode(d))
        elif kind == 3:
            out[name] = json.loads(d.decode())
        elif kind == 4:
            out[name] = d.decode().split("\0") if d else []
        else:
            raise ValueError(f"άγνωστο kind {kind}")
    return out


# ---------------- γεωμετρία ----------------
R = 6371008.8


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=np.float64)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


def morton(x, y):
    """Z-order κλειδί για int32 x,y (σε 1e-6°) -> καλύτερη τοπικότητα/συμπίεση."""
    def spread(v):
        v = (np.asarray(v, dtype=np.int64) - np.int64(-(1 << 31))).astype(np.uint64) >> np.uint64(5)
        v &= np.uint64(0x7FFFFFF)
        r = np.zeros_like(v)
        for b in range(27):
            r |= ((v >> np.uint64(b)) & np.uint64(1)) << np.uint64(2 * b)
        return r
    return spread(x) | (spread(y) << np.uint64(1))


class Polygon:
    """Even-odd point-in-polygon για πολλά δαχτυλίδια (νησιά/τρύπες), μόνο numpy.
    Raster για τα εσωτερικά κελιά + ακριβής έλεγχος στα κελιά που αγγίζει το όριο."""

    def __init__(self, rings, res=0.0005):
        xs1, ys1, xs2, ys2 = [], [], [], []
        for r in rings:
            r = np.asarray(r, dtype=np.float64)
            if len(r) < 3:
                continue
            if (r[0] != r[-1]).any():
                r = np.vstack([r, r[:1]])
            xs1.append(r[:-1, 0]); ys1.append(r[:-1, 1]); xs2.append(r[1:, 0]); ys2.append(r[1:, 1])
        if not xs1:
            raise ValueError("κενό πολύγωνο")
        self.x1, self.y1 = np.concatenate(xs1), np.concatenate(ys1)
        self.x2, self.y2 = np.concatenate(xs2), np.concatenate(ys2)
        self.res = res
        self.x0 = min(self.x1.min(), self.x2.min()) - res
        self.y0 = min(self.y1.min(), self.y2.min()) - res
        self.nx = int((max(self.x1.max(), self.x2.max()) - self.x0) / res) + 2
        self.ny = int((max(self.y1.max(), self.y2.max()) - self.y0) / res) + 2
        # edges -> γραμμές raster
        r0 = np.floor((np.minimum(self.y1, self.y2) - self.y0) / res).astype(np.int64)
        r1 = np.floor((np.maximum(self.y1, self.y2) - self.y0) / res).astype(np.int64)
        cnt = r1 - r0 + 1
        eid = np.repeat(np.arange(cnt.size), cnt)
        off = np.arange(eid.size) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        row = r0[eid] + off
        o = np.argsort(row, kind="stable")
        self.row_e = eid[o]
        self.row_ptr = np.searchsorted(row[o], np.arange(self.ny + 1))
        # κελιά που αγγίζει το όριο: δειγματοληψία κάθε ακμής
        L = np.hypot(self.x2 - self.x1, self.y2 - self.y1)
        ns = np.maximum(1, np.ceil(L / (res / 3)).astype(np.int64)) + 1
        e = np.repeat(np.arange(ns.size), ns)
        t = (np.arange(e.size) - np.repeat(np.cumsum(ns) - ns, ns)) / (ns[e] - 1)
        px = self.x1[e] + t * (self.x2[e] - self.x1[e])
        py = self.y1[e] + t * (self.y2[e] - self.y1[e])
        self.border = np.zeros((self.ny, self.nx), dtype=bool)
        cx = np.clip(((px - self.x0) / res).astype(np.int64), 0, self.nx - 1)
        cy = np.clip(((py - self.y0) / res).astype(np.int64), 0, self.ny - 1)
        self.border[cy, cx] = True
        # γέμισμα με scanline στο κέντρο κάθε γραμμής
        self.mask = np.zeros((self.ny, self.nx), dtype=bool)
        xc = self.x0 + (np.arange(self.nx) + 0.5) * res
        for r in range(self.ny):
            ed = self.row_e[self.row_ptr[r]:self.row_ptr[r + 1]]
            if ed.size == 0:
                continue
            yc = self.y0 + (r + 0.5) * res
            cr = self._cross(ed, yc)
            if cr.size:
                self.mask[r] = (np.searchsorted(np.sort(cr), xc) & 1).astype(bool)

    def _cross(self, ed, y):
        y1, y2 = self.y1[ed], self.y2[ed]
        m = (y1 > y) != (y2 > y)
        ed = ed[m]
        y1, y2 = y1[m], y2[m]
        return self.x1[ed] + (y - y1) * (self.x2[ed] - self.x1[ed]) / (y2 - y1)

    def contains(self, lon, lat):
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        out = np.zeros(lon.shape, dtype=bool)
        cx = np.floor((lon - self.x0) / self.res).astype(np.int64)
        cy = np.floor((lat - self.y0) / self.res).astype(np.int64)
        ok = (cx >= 0) & (cx < self.nx) & (cy >= 0) & (cy < self.ny)
        idx = np.flatnonzero(ok)
        cxo, cyo = cx[idx], cy[idx]
        b = self.border[cyo, cxo]
        out[idx[~b]] = self.mask[cyo[~b], cxo[~b]]
        bi = idx[b]
        if bi.size:
            rows = cy[bi]
            o = np.argsort(rows, kind="stable")
            bi, rows = bi[o], rows[o]
            cuts = np.flatnonzero(np.diff(rows)) + 1
            for grp in np.split(np.arange(bi.size), cuts):
                r = rows[grp[0]]
                ed = self.row_e[self.row_ptr[r]:self.row_ptr[r + 1]]
                pts = bi[grp]
                px, py = lon[pts], lat[pts]
                for s in range(0, pts.size, 4096):
                    X = px[s:s + 4096, None]
                    Y = py[s:s + 4096, None]
                    y1, y2 = self.y1[ed][None], self.y2[ed][None]
                    x1, x2 = self.x1[ed][None], self.x2[ed][None]
                    m = (y1 > Y) != (y2 > Y)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        xc = x1 + (Y - y1) * (x2 - x1) / (y2 - y1)
                    out[pts[s:s + 4096]] = ((m & (X < xc)).sum(1) & 1).astype(bool)
        return out
