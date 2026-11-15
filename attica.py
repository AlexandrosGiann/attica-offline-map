"""Χάρτης Αττικής ως γράφος + Dijkstra (αυτοκίνητο, πεζός, ποδήλατο, ΜΜΜ).

    python attica.py info
    python attica.py search "Σύνταγμα"
    python attica.py route "Σύνταγμα" "Πειραιάς" --mode transit [--geojson r.geojson]
    python attica.py route "37.97,23.73" "Κηφισιά" --mode car [--metric dist]
    python attica.py nearby "Μοναστηράκι" --radius 300 [--tag amenity=cafe]
    python attica.py lines "X95"
    python attica.py stops "Σύνταγμα"
    python attica.py wv "Piraeus"
    python attica.py export out.geojson --what roads|pois|lines|stops|boundary

Μοναδική εξάρτηση: numpy. Αρχείο δεδομένων: attica.amap (δίπλα στο script).
"""
import argparse
import heapq
import json
import math
import os
import re
import sys
import time
import unicodedata

import numpy as np

import amap
from amap import haversine

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MAP = os.path.join(HERE, "attica.amap")

WALK = 5.0 / 3.6      # m/s
BIKE = 15.0 / 3.6
FERRY_WAIT = 1800      # s, αναμονή για πλοίο
ACCESS_MAX = 500       # m, μέγιστη απόσταση στάσης από δίκτυο πεζών
# τύπος γραμμής -> (km/h, αναμονή s αν δεν υπάρχει interval)
TRANSIT_PARAMS = {"bus": (18, 600), "trolleybus": (16, 600), "coach": (45, 1800),
                  "tram": (18, 420), "subway": (33, 240), "light_rail": (30, 300),
                  "monorail": (30, 300), "train": (55, 900), "ferry": (30, 1800),
                  "funicular": (10, 600), "share_taxi": (25, 600)}
TYPE_EL = {"bus": "Λεωφορείο", "trolleybus": "Τρόλεϊ", "coach": "Υπεραστικό", "tram": "Τραμ",
           "subway": "Μετρό", "light_rail": "Ηλεκτρικός", "monorail": "Monorail",
           "train": "Τρένο/Προαστιακός", "ferry": "Πλοίο", "funicular": "Τελεφερίκ",
           "share_taxi": "Ταξί"}


def norm(s):
    s = unicodedata.normalize("NFD", s.casefold().replace("ς", "σ"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^\w]+", " ", s).strip()


def fmt_t(s):
    s = int(round(s))
    h, m = divmod(s // 60, 60)
    return f"{h}ω {m}λ" if h else f"{max(m, 1) if s >= 30 else 0}λ" if s >= 30 else f"{s}δ"


def fmt_d(m):
    return f"{m / 1000:.1f} km" if m >= 1000 else f"{m:.0f} m"


def _segcum(d, n, start):
    """cumsum ανά τμήμα: x[i] = start_e + sum(d[s_e..i])"""
    if d.size == 0:
        return d.astype(np.int64)
    cs = np.cumsum(d)
    off = np.cumsum(n) - n
    base = np.where(off > 0, cs[np.maximum(off - 1, 0)], 0)
    base = np.where(n > 0, base, 0)
    return cs - np.repeat(base - start, n)


class Map:
    def __init__(self, path=DEFAULT_MAP):
        if not os.path.isfile(path):
            sys.exit(f"Δεν βρέθηκε ο χάρτης: {path} (τρέξε build.py)")
        t = time.time()
        d = amap.load(path)
        self.file_size = os.path.getsize(path)
        self.meta = d["meta"]
        sc = float(self.meta["scale"])
        self.S = d["str"]
        self.F = self.meta["flags"]
        self.HW = self.meta["highway_classes"]
        # κορυφές
        self.vlon = np.cumsum(d["v.dx"]) / sc
        self.vlat = np.cumsum(d["v.dy"]) / sc
        self.vosm = np.cumsum(d["v.osm"])
        # ακμές
        dv = d["e.v"]
        self.ev = np.cumsum(d["e.u"] + dv)
        self.eu = self.ev - dv
        self.elen = d["e.len"] / 10.0
        self.ecls = d["e.cls"]
        self.eflags = d["e.flags"]
        self.espeed = d["e.speed"]
        self.eway = np.cumsum(d["e.way"])
        self.eni = d["e.ni"]
        self.eoff = np.concatenate([[0], np.cumsum(self.eni)])
        vx = np.cumsum(d["v.dx"])
        vy = np.cumsum(d["v.dy"])
        self.glon = _segcum(d["g.dx"], self.eni, vx[self.eu]) / sc
        self.glat = _segcum(d["g.dy"], self.eni, vy[self.eu]) / sc
        # ways
        self.way_osm = np.cumsum(d["way.osm"])
        self.way_nt = d["way.nt"]
        self.way_off = np.concatenate([[0], np.cumsum(self.way_nt)]) * 2
        self.way_kv = d["way.kv"]
        # POI
        self.plon = np.cumsum(d["p.dx"]) / sc
        self.plat = np.cumsum(d["p.dy"]) / sc
        self.posm = d["p.osm"]
        self.parea = d["p.area"]
        self.p_nt = d["p.nt"]
        self.p_off = np.concatenate([[0], np.cumsum(self.p_nt)]) * 2
        self.p_kv = d["p.kv"]
        # όριο
        n = d["bnd.n"]
        bx = np.cumsum(d["bnd.dx"]) / sc   # μία συνεχής σειρά διαφορών για όλα τα δαχτυλίδια
        by = np.cumsum(d["bnd.dy"]) / sc
        o = np.concatenate([[0], np.cumsum(n)])
        self.boundary = [np.column_stack([bx[o[i]:o[i + 1]], by[o[i]:o[i + 1]]]) for i in range(n.size)]
        # ΜΜΜ
        tr = d["transit"]
        self.routes = tr["routes"]
        self.masters = tr["masters"]
        self.stop_areas = tr["stop_areas"]
        n = d["rs.n"]
        o = np.concatenate([[0], np.cumsum(n)])
        sx = _segcum(d["rs.dx"], n, np.zeros(n.size, np.int64)) / sc
        sy = _segcum(d["rs.dy"], n, np.zeros(n.size, np.int64)) / sc
        for i, r in enumerate(self.routes):
            r["shape"] = (sx[o[i]:o[i + 1]], sy[o[i]:o[i + 1]])
        self.wv = d["wv"]
        self._graphs = {}
        self._index = None
        self.load_s = time.time() - t

    # ---------- tags ----------
    def way_tags(self, w):
        kv = self.way_kv[self.way_off[w]:self.way_off[w + 1]]
        return {self.S[kv[i]]: self.S[kv[i + 1]] for i in range(0, kv.size, 2)}

    def poi_tags(self, p):
        kv = self.p_kv[self.p_off[p]:self.p_off[p + 1]]
        return {self.S[kv[i]]: self.S[kv[i + 1]] for i in range(0, kv.size, 2)}

    def poi_name(self, p, t=None, _near=True):
        t = t or self.poi_tags(p)
        n = t.get("name") or t.get("name:el") or t.get("name:en") or \
            (f"{t.get('addr:street', '')} {t.get('addr:housenumber', '')}".strip())
        if n:
            return n
        if _near:  # ανώνυμη στάση: όνομα της κοντινότερης ονομασμένης στάσης/σταθμού (<150 m)
            d = haversine(self.plat[p], self.plon[p], self.plat, self.plon)
            for i in np.argsort(d)[1:30]:
                if d[i] > 150:
                    break
                ti = self.poi_tags(i)
                if ti.get("name") and self._is_stop_tags(i):
                    return ti["name"]
        return t.get("ref") or f"POI {p}"

    def edge_coords(self, e, reverse=False):
        a, b = self.eoff[e], self.eoff[e + 1]
        lon = np.concatenate([[self.vlon[self.eu[e]]], self.glon[a:b], [self.vlon[self.ev[e]]]])
        lat = np.concatenate([[self.vlat[self.eu[e]]], self.glat[a:b], [self.vlat[self.ev[e]]]])
        if reverse:
            lon, lat = lon[::-1], lat[::-1]
        return lon, lat

    # ---------- γράφοι ----------
    def graph(self, mode, metric="time"):
        key = (mode, metric)
        if key in self._graphs:
            return self._graphs[key]
        F, fl, L = self.F, self.eflags, self.elen
        ferry = (fl & F["ferry"]) > 0
        if mode == "car":
            fw, bw = (fl & F["car_f"]) > 0, (fl & F["car_b"]) > 0
            w = L / (np.maximum(self.espeed, 5) / 3.6)
        elif mode == "bike":
            fw, bw = (fl & F["bike_f"]) > 0, (fl & F["bike_b"]) > 0
            w = L / np.where(ferry, np.maximum(self.espeed, 5) / 3.6, BIKE)
        else:  # foot / transit
            fw = bw = (fl & F["foot"]) > 0
            steps = self.ecls == self.HW.index("steps")
            w = L / np.where(ferry, np.maximum(self.espeed, 5) / 3.6, np.where(steps, WALK / 1.6, WALK))
        w = np.where(ferry, w + FERRY_WAIT, w)
        if metric == "dist":
            w = L.copy()
        E = np.arange(self.eu.size)
        src = np.concatenate([self.eu[fw], self.ev[bw]])
        dst = np.concatenate([self.ev[fw], self.eu[bw]])
        wt = np.concatenate([w[fw], w[bw]])
        eid = np.concatenate([E[fw], -E[bw] - 1])  # αρνητικό = αντίστροφη φορά
        g = {"mode": mode, "nv": self.vlon.size, "src": src, "dst": dst, "w": wt, "eid": eid,
             "fw": fw, "bw": bw, "ew": w, "extra": None}
        # ασθενώς συνεκτικές συνιστώσες (union-find): το snap αγνοεί μικρές νησίδες
        par = list(range(self.vlon.size))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        m = fw | bw
        for a, b in zip(self.eu[m].tolist(), self.ev[m].tolist()):
            ra, rb = find(a), find(b)
            if ra != rb:
                par[ra] = rb
        root = np.array([find(x) for x in range(self.vlon.size)])
        _, inv, cnt = np.unique(root, return_inverse=True, return_counts=True)
        g["big"] = cnt[inv] >= 200
        if mode == "transit":
            self._add_transit(g)
        self._finish(g)
        self._graphs[key] = g
        return g

    def _finish(self, g):
        n = g["nv"] + (g["extra"]["n"] if g["extra"] else 0)
        o = np.argsort(g["src"], kind="stable")
        for k in ("src", "dst", "w", "eid"):
            g[k] = g[k][o]
        g["ptr"] = np.searchsorted(g["src"], np.arange(n + 1)).tolist()
        g["dst_l"] = g["dst"].tolist()
        g["w_l"] = g["w"].tolist()
        g["deg"] = np.diff(np.array(g["ptr"]))

    def _add_transit(self, g):
        fw = g["fw"]
        okv = np.zeros(self.vlon.size, bool)
        fb = fw & g["big"][self.eu]
        okv[self.eu[fb]] = True
        okv[self.ev[fb]] = True
        cand = np.flatnonzero(okv)
        cl, ca = self.vlon[cand], self.vlat[cand]
        nv = self.vlon.size
        src, dst, w, eid = [], [], [], []
        info = []   # ανά ride-κορυφή: (route, θέση, poi)
        cache = {}

        def access(p):
            if p not in cache:
                lo, la = self.plon[p], self.plat[p]
                dx = (cl - lo) * math.cos(math.radians(la))
                i = int(np.argmin(dx * dx + (ca - la) ** 2))
                dd = float(haversine(la, lo, ca[i], cl[i]))
                cache[p] = (int(cand[i]), dd) if dd <= ACCESS_MAX else (None, dd)
            return cache[p]

        for ri, r in enumerate(self.routes):
            t = r["tags"]
            kind = t.get("route", "bus")
            spd, wait = TRANSIT_PARAMS.get(kind, (18, 600))
            iv = t.get("interval")
            if iv:
                m = re.match(r"^(?:(\d+):)?(\d+)", iv)
                if m:
                    mins = int(m.group(1) or 0) * 60 + int(m.group(2))
                    if 0 < mins < 300:
                        wait = mins * 60 / 2
            # στάσεις: συγχώνευση στάσης/πλατφόρμας της ίδιας στάσης
            seq = []
            for p, role in r["stops"]:
                if seq and haversine(self.plat[p], self.plon[p], self.plat[seq[-1]], self.plon[seq[-1]]) < 80:
                    if role.startswith("platform") and not self._is_platform(seq[-1]):
                        seq[-1] = p
                    continue
                seq.append(p)
            seq = [p for p in seq if access(p)[0] is not None]
            if len(seq) < 2:
                continue
            # απόσταση κατά μήκος της διαδρομής
            shx, shy = r["shape"]
            cum = None
            if shx.size > 1:
                cum = np.concatenate([[0], np.cumsum(haversine(shy[:-1], shx[:-1], shy[1:], shx[1:]))])
                k = []
                for p in seq:
                    dx = (shx - self.plon[p]) * math.cos(math.radians(self.plat[p]))
                    k.append(int(np.argmin(dx * dx + (shy - self.plat[p]) ** 2)))
                k = np.array(k)
            both = t.get("public_transport:version") == "1"
            base = nv + len(info)
            for j, p in enumerate(seq):
                info.append((ri, j, p))
                rv = base + j
                av, ad = access(p)
                src += [av, rv]
                dst += [rv, av]
                w += [wait + ad / WALK, ad / WALK]
                eid += [-(10 ** 12), -(10 ** 12)]
            for j in range(len(seq) - 1):
                a, b = seq[j], seq[j + 1]
                hd = float(haversine(self.plat[a], self.plon[a], self.plat[b], self.plon[b]))
                d = hd * 1.25
                if cum is not None and k[j + 1] > k[j]:
                    sd = float(cum[k[j + 1]] - cum[k[j]])
                    if hd * 0.9 <= sd <= hd * 4:
                        d = sd
                tt = d / (spd / 3.6) + (20 if kind in ("bus", "trolleybus", "tram") else 30)
                src.append(base + j); dst.append(base + j + 1); w.append(tt); eid.append(-(10 ** 12))
                if both:
                    src.append(base + j + 1); dst.append(base + j); w.append(tt); eid.append(-(10 ** 12))
        g["extra"] = {"n": len(info), "info": info}
        g["src"] = np.concatenate([g["src"], np.array(src, np.int64)])
        g["dst"] = np.concatenate([g["dst"], np.array(dst, np.int64)])
        g["w"] = np.concatenate([g["w"], np.array(w, float)])
        g["eid"] = np.concatenate([g["eid"], np.array(eid, np.int64)])

    def _is_platform(self, p):
        t = self.poi_tags(p)
        return t.get("public_transport") == "platform" or t.get("highway") == "bus_stop"

    # ---------- snapping ----------
    def snap(self, lat, lon, g):
        """Κοντινότερο σημείο του δικτύου που επιτρέπεται στο mode.
        -> λίστα (κορυφή, αρχικό κόστος, κορυφή-ως-στόχος κόστος), περιγραφή"""
        fw, bw = g["fw"], g["bw"]
        ok = (fw | bw) & g["big"][self.eu]
        okv = np.zeros(self.vlon.size, bool)
        okv[self.eu[ok]] = True
        okv[self.ev[ok]] = True
        if not okv.any():  # χωρίς μεγάλη συνιστώσα: δέξου οτιδήποτε
            ok = fw | bw
            okv[self.eu[ok]] = True
            okv[self.ev[ok]] = True
        c = math.cos(math.radians(lat))
        # κορυφές
        vi = np.flatnonzero(okv)
        dv = ((self.vlon[vi] - lon) * c) ** 2 + (self.vlat[vi] - lat) ** 2
        best_v = vi[np.argmin(dv)] if vi.size else -1
        bdv = dv.min() if vi.size else np.inf
        # ενδιάμεσα σημεία ακμών
        eidx = np.repeat(np.arange(self.eu.size), self.eni)
        gm = ok[eidx]
        gi = np.flatnonzero(gm)
        dg = ((self.glon[gi] - lon) * c) ** 2 + (self.glat[gi] - lat) ** 2
        if gi.size and dg.min() < bdv:
            j = gi[np.argmin(dg)]
            e = int(eidx[j])
            lo, la = self.edge_coords(e)
            seg = haversine(la[:-1], lo[:-1], la[1:], lo[1:])
            cum = np.concatenate([[0], np.cumsum(seg)])
            s = cum[j - self.eoff[e] + 1]
            Lt = cum[-1]
            fr = float(s / Lt) if Lt > 0 else 0.0
            snapd = float(haversine(lat, lon, self.glat[j], self.glon[j]))
            return ("e", e, fr), snapd
        snapd = float(haversine(lat, lon, self.vlat[best_v], self.vlon[best_v]))
        return ("v", int(best_v), 0.0), snapd

    def _ends(self, g, sn, as_target):
        """Κόστη από σημείο προς τις κορυφές (πηγή) ή από κορυφές προς σημείο (στόχος)."""
        if sn[0] == "v":
            return {sn[1]: 0.0}
        e, fr = sn[1], sn[2]
        ew = float(g["ew"][e])
        u, v = int(self.eu[e]), int(self.ev[e])
        out = {}
        if as_target:   # u->σημείο θέλει κανονική φορά, v->σημείο αντίστροφη
            if g["fw"][e]:
                out[u] = ew * fr
            if g["bw"][e]:
                out[v] = min(out.get(v, math.inf), ew * (1 - fr))
        else:           # σημείο->v κανονική, σημείο->u αντίστροφη
            if g["fw"][e]:
                out[v] = ew * (1 - fr)
            if g["bw"][e]:
                out[u] = min(out.get(u, math.inf), ew * fr)
        return out

    # ---------- Dijkstra ----------
    def dijkstra(self, g, sources, targets):
        """sources: {κορυφή: κόστος}, targets: {κορυφή: επιπλέον κόστος}.
        Κλασικός Dijkstra με binary heap, σταματά όταν κανένας στόχος δεν μπορεί να βελτιωθεί."""
        ptr, dst, w = g["ptr"], g["dst_l"], g["w_l"]
        dist = {}
        pred = {}
        h = []
        for v, c in sources.items():
            if c < dist.get(v, math.inf):
                dist[v] = c
                pred[v] = -1
                heapq.heappush(h, (c, v))
        best, best_t = math.inf, None
        done = set()
        settled = 0
        while h:
            c, v = heapq.heappop(h)
            if v in done:
                continue
            if c >= best:
                break
            done.add(v)
            settled += 1
            if v in targets and c + targets[v] < best:
                best, best_t = c + targets[v], v
            for i in range(ptr[v], ptr[v + 1]):
                x = dst[i]
                nc = c + w[i]
                if nc < dist.get(x, math.inf):
                    dist[x] = nc
                    pred[x] = i
                    heapq.heappush(h, (nc, x))
        if best_t is None:
            return None
        path = []
        v = best_t
        while pred[v] != -1:
            i = pred[v]
            path.append(i)
            v = g["src"][i]
        path.reverse()
        return best, best_t, v, path, settled

    # ---------- αναζήτηση ----------
    def index(self):
        if self._index is not None:
            return self._index
        keys = ("name", "name:el", "name:en", "alt_name", "official_name", "short_name",
                "old_name", "int_name", "loc_name", "ref")
        names, kind, ref = [], [], []
        kid = {self.S[i]: i for i in range(len(self.S))}
        want = {kid[k] for k in keys if k in kid}
        kvs = self.p_kv
        own = np.repeat(np.arange(self.p_nt.size), self.p_nt)
        kk = kvs[0::2]
        sel = np.isin(kk, list(want))
        for p, vi, ki in zip(own[sel].tolist(), kvs[1::2][sel].tolist(), kk[sel].tolist()):
            if self.S[ki] == "ref" and not self._is_stop_tags(p):
                continue
            names.append(norm(self.S[vi])); kind.append("p"); ref.append(p)
        for i, a in enumerate(self.wv["articles"]):
            names.append(norm(a["title"])); kind.append("a"); ref.append(i)
        for i, l in enumerate(self.wv["listings"]):
            if l.get("name"):
                names.append(norm(l["name"])); kind.append("l"); ref.append(i)
        self._index = (names, kind, ref)
        return self._index

    def _is_stop_tags(self, p):
        t = self.poi_tags(p)
        return "public_transport" in t or t.get("highway") == "bus_stop" or "railway" in t

    def _rank(self, kind, ref):
        if kind == "a":
            return 0
        if kind == "l":
            return 6
        t = self.poi_tags(ref)
        pl = t.get("place")
        if pl in ("city", "town", "municipality"):
            return 1
        if t.get("railway") == "station" or t.get("public_transport") == "station" or \
                t.get("aeroway") == "aerodrome" or t.get("amenity") == "ferry_terminal":
            return 2
        if pl in ("suburb", "village", "quarter", "neighbourhood", "island"):
            return 3
        if t.get("tourism") or t.get("historic") or t.get("amenity") in ("university", "hospital"):
            return 4
        return 5

    def search(self, text, limit=10):
        m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", text)
        if m:
            return [{"kind": "coord", "name": text, "lat": float(m.group(1)), "lon": float(m.group(2))}]
        qn = norm(text)
        if not qn:
            return []
        names, kind, ref = self.index()
        hits = []
        for i, n in enumerate(names):
            if n == qn:
                tier = 0
            elif n.startswith(qn + " ") or n.endswith(" " + qn):
                tier = 1
            elif qn in n:
                tier = 2
            else:
                continue
            hits.append((tier, self._rank(kind[i], ref[i]), len(n), kind[i], ref[i]))
        hits.sort()
        out, seen = [], set()
        for tier, rk, _, k, r in hits:
            if (k, r) in seen:
                continue
            seen.add((k, r))
            out.append(self.describe(k, r))
            if len(out) >= limit:
                break
        return out

    def describe(self, k, r):
        if k == "p":
            t = self.poi_tags(r)
            typ = next((f"{x}={t[x]}" for x in ("place", "railway", "public_transport", "amenity", "shop",
                                                  "tourism", "historic", "leisure", "highway", "natural",
                                                  "office", "aeroway") if x in t), "")
            return {"kind": "poi", "id": r, "name": self.poi_name(r, t), "type": typ,
                    "lat": float(self.plat[r]), "lon": float(self.plon[r]), "tags": t}
        if k == "a":
            a = self.wv["articles"][r]
            return {"kind": "article", "id": r, "name": a["title"], "type": "wikivoyage",
                    "lat": a["lat"], "lon": a["lon"]}
        lst = self.wv["listings"][r]
        return {"kind": "listing", "id": r, "name": lst["name"], "type": f"wikivoyage:{lst.get('sec', '')}",
                "lat": lst["lat"], "lon": lst["lon"], "tags": lst}

    def resolve(self, text):
        r = self.search(text, 1)
        if not r:
            sys.exit(f"Δεν βρέθηκε: {text}")
        return r[0]

    # ---------- διαδρομή ----------
    def route(self, a, b, mode="car", metric="time"):
        g = self.graph(mode, metric)
        sa, da = self.snap(a["lat"], a["lon"], g)
        sb, db = self.snap(b["lat"], b["lon"], g)
        sources = self._ends(g, sa, False)
        targets = self._ends(g, sb, True)
        # μέση ταχύτητα πρόσβασης από το σημείο στο δίκτυο (πεζός) για να μετρά το snap
        t0 = time.time()
        res = self.dijkstra(g, sources, targets)
        dt = time.time() - t0
        if res is None:
            return None
        cost, vt, vs, path, settled = res
        return self._explain(g, path, cost, a, b, da, db, settled, dt, metric)

    def _explain(self, g, path, cost, a, b, da, db, settled, dt, metric):
        nv = g["nv"]
        steps, lon, lat = [], [], []
        dist_total = 0.0
        cur = None

        def flush():
            if cur:
                steps.append(cur)

        for i in path:
            s, d, e = int(g["src"][i]), int(g["dst"][i]), int(g["eid"][i])
            if e >= 0 or (e < 0 and e > -(10 ** 12)):
                ed = e if e >= 0 else -e - 1
                lo, la = self.edge_coords(ed, reverse=e < 0)
                lon.extend(lo.tolist()); lat.extend(la.tolist())
                dist_total += self.elen[ed]
                wt = self.way_tags(int(self.eway[ed]))
                name = wt.get("name") or wt.get("ref") or (
                    "πλοίο" if self.eflags[ed] & self.F["ferry"] else self.HW[self.ecls[ed]])
                kind = "walk" if g["mode"] in ("foot", "transit") else g["mode"]
                if cur and cur["type"] == kind and cur["name"] == name:
                    cur["dist"] += self.elen[ed]; cur["time"] += g["w"][i]
                else:
                    flush()
                    cur = {"type": kind, "name": name, "dist": float(self.elen[ed]), "time": float(g["w"][i])}
            else:
                info = g["extra"]["info"]
                if s < nv and d >= nv:          # επιβίβαση
                    ri, j, p = info[d - nv]
                    flush()
                    r = self.routes[ri]
                    cur = {"type": "ride", "route": ri, "from": p, "to": p, "stops": 0,
                           "time": float(g["w"][i]), "dist": 0.0,
                           "name": self.route_label(r)}
                    lon.append(float(self.plon[p])); lat.append(float(self.plat[p]))
                elif s >= nv and d >= nv:       # μετακίνηση με το μέσο
                    p = info[d - nv][2]
                    q_ = info[s - nv][2]
                    cur["stops"] += 1
                    cur["to"] = p
                    cur["time"] += float(g["w"][i])
                    cur["dist"] += float(haversine(self.plat[q_], self.plon[q_], self.plat[p], self.plon[p]))
                    lon.append(float(self.plon[p])); lat.append(float(self.plat[p]))
                else:                            # αποβίβαση
                    cur["time"] += float(g["w"][i])
                    flush()
                    cur = None
        flush()
        return {"cost": cost, "metric": metric, "mode": g["mode"], "from": a, "to": b,
                "snap": (da, db), "steps": steps, "dist": dist_total, "lon": lon, "lat": lat,
                "settled": settled, "dijkstra_s": dt}

    def route_label(self, r):
        t = r["tags"]
        kind = TYPE_EL.get(t.get("route"), t.get("route", ""))
        ref = t.get("ref", "")
        nm = t.get("name", "")
        return re.sub(r"\s+", " ", f"{kind} {ref} ({nm})".replace(" ()", "")).strip()


# ---------------- CLI ----------------
def print_route(M, r):
    if r is None:
        print("Δεν βρέθηκε διαδρομή (μη συνδεδεμένα σημεία για αυτό το μέσο).")
        return
    print(f"Από: {r['from']['name']}  ->  Προς: {r['to']['name']}   [{r['mode']}]")
    if r["snap"][0] > 300 or r["snap"][1] > 300:
        print(f"  ! απόσταση από δίκτυο: {fmt_d(r['snap'][0])} / {fmt_d(r['snap'][1])}")
    tt = 0.0
    for s in r["steps"]:
        if s["type"] == "ride":
            print(f"  🚌 {s['name']}: {M.poi_name(s['from'])} -> {M.poi_name(s['to'])}, "
                  f"{s['stops']} στάσεις, ~{fmt_t(s['time'])} (με αναμονή)")
        else:
            verb = {"walk": "Περπάτημα", "car": "Οδήγηση", "bike": "Ποδήλατο"}[s["type"]]
            print(f"  {verb}: {s['name']} {fmt_d(s['dist'])} ~{fmt_t(s['time'])}")
        tt += s["time"]
    if r["metric"] == "dist":
        print(f"Σύνολο: {fmt_d(r['cost'])}")
    else:
        print(f"Σύνολο: ~{fmt_t(r['cost'])}, {fmt_d(r['dist'])} οδικά")
    print(f"(Dijkstra: {r['settled']} κορυφές σε {r['dijkstra_s'] * 1000:.0f} ms)")


def geojson_feature(lon, lat, props):
    return {"type": "Feature", "properties": props,
            "geometry": {"type": "LineString", "coordinates": [[round(x, 6), round(y, 6)] for x, y in zip(lon, lat)]}}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Χάρτης Αττικής (γράφος + Dijkstra)")
    ap.add_argument("--map", default=DEFAULT_MAP)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("info")
    s = sp.add_parser("search"); s.add_argument("text"); s.add_argument("-n", type=int, default=10)
    s = sp.add_parser("route"); s.add_argument("a"); s.add_argument("b")
    s.add_argument("--mode", choices=["car", "foot", "bike", "transit"], default="car")
    s.add_argument("--metric", choices=["time", "dist"], default="time")
    s.add_argument("--geojson")
    s = sp.add_parser("nearby"); s.add_argument("place"); s.add_argument("--radius", type=float, default=300)
    s.add_argument("--tag"); s.add_argument("-n", type=int, default=20)
    s = sp.add_parser("lines"); s.add_argument("text")
    s = sp.add_parser("stops"); s.add_argument("place"); s.add_argument("--radius", type=float, default=300)
    s = sp.add_parser("wv"); s.add_argument("title")
    s = sp.add_parser("export"); s.add_argument("out")
    s.add_argument("--what", choices=["roads", "pois", "lines", "stops", "boundary"], default="lines")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    M = Map(a.map)

    if a.cmd == "info":
        c = M.meta["counts"]
        print(f"{M.meta['name']} — αρχείο {M.file_size / 1e6:.2f} MB, φόρτωση {M.load_s:.1f}s")
        print(f"κορυφές {c['vertices']}, ακμές {c['edges']}, σημεία γεωμετρίας {M.glon.size}, δρόμοι {M.way_osm.size}")
        print(f"POI {c['pois']}, γραμμές ΜΜΜ {c['routes']}, Wikivoyage άρθρα {c['articles']}, listings {c['listings']}")
        from collections import Counter
        print("γραμμές ανά τύπο:", dict(Counter(r["tags"].get("route") for r in M.routes)))
        print("πηγές:", "; ".join(M.meta["sources"]))
    elif a.cmd == "search":
        for r in M.search(a.text, a.n):
            print(f"{r['name']}  [{r['type']}]  {r['lat']:.5f},{r['lon']:.5f}")
    elif a.cmd == "route":
        A, B = M.resolve(a.a), M.resolve(a.b)
        r = M.route(A, B, a.mode, a.metric)
        print_route(M, r)
        if r and a.geojson:
            fc = {"type": "FeatureCollection", "features": [geojson_feature(r["lon"], r["lat"], {
                "mode": a.mode, "cost": r["cost"], "from": A["name"], "to": B["name"]})]}
            with open(a.geojson, "w", encoding="utf-8") as f:
                json.dump(fc, f, ensure_ascii=False)
            print("GeoJSON:", a.geojson)
    elif a.cmd in ("nearby", "stops"):
        P = M.resolve(a.place)
        d = haversine(P["lat"], P["lon"], M.plat, M.plon)
        idx = np.flatnonzero(d <= a.radius)
        idx = idx[np.argsort(d[idx])]
        n = 0
        for i in idx:
            t = M.poi_tags(i)
            if a.cmd == "stops":
                if not M._is_stop_tags(i):
                    continue
            elif a.tag:
                k, _, v = a.tag.partition("=")
                if k not in t or (v and t[k] != v):
                    continue
            elif "name" not in t:
                continue
            lines = [M.route_label(r) for r in M.routes if any(p == i for p, _ in r["stops"])]
            extra = ("  γραμμές: " + ", ".join(sorted(set(lines)))) if lines else ""
            print(f"{d[i]:5.0f} m  {M.poi_name(i, t)}  [{M.describe('p', i)['type']}]{extra}")
            n += 1
            if a.cmd == "nearby" and n >= a.n:
                break
    elif a.cmd == "lines":
        # ελληνικά/λατινικά ομόγραφα (Χ95 = X95, Α1 = A1)
        look = str.maketrans("αβεζηικμνορτυχ", "abezhikmnoptyx")
        qn = norm(a.text).translate(look)
        for r in M.routes:
            t = r["tags"]
            if qn in norm(t.get("ref", "")).translate(look) or qn in norm(t.get("name", "")).translate(look):
                print(M.route_label(r), "|", t.get("operator", ""), "|", t.get("from", ""), "->", t.get("to", ""))
                print("   ", " → ".join(M.poi_name(p) for p, role in r["stops"] if role.startswith("platform") or role == "stop"))
    elif a.cmd == "wv":
        qn = norm(a.title)
        arts = [x for x in M.wv["articles"] if norm(x["title"]) == qn] or \
               [x for x in M.wv["articles"] if qn in norm(x["title"])]
        if not arts:
            sys.exit("Δεν βρέθηκε άρθρο")
        x = arts[0]
        print(f"# {x['title']} ({x['lat']}, {x['lon']})")
        for h, t in x["sections"]:
            if h:
                print("\n" + h)
            print(t)
    elif a.cmd == "export":
        feats = []
        if a.what == "roads":
            for e in range(M.eu.size):
                lo, la = M.edge_coords(e)
                feats.append(geojson_feature(lo, la, {"highway": M.HW[M.ecls[e]], "len": float(M.elen[e]),
                                                       "way": int(M.way_osm[M.eway[e]])}))
        elif a.what == "lines":
            for r in M.routes:
                if r["shape"][0].size:
                    feats.append(geojson_feature(r["shape"][0], r["shape"][1], r["tags"]))
        elif a.what == "boundary":
            for ring in M.boundary:
                feats.append(geojson_feature(ring[:, 0], ring[:, 1], {"name": M.meta["name"]}))
        else:
            for i in range(M.plon.size):
                if a.what == "stops" and not M._is_stop_tags(i):
                    continue
                feats.append({"type": "Feature", "properties": M.poi_tags(i), "geometry": {
                    "type": "Point", "coordinates": [round(float(M.plon[i]), 6), round(float(M.plat[i]), 6)]}})
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
        print(a.out, len(feats))


if __name__ == "__main__":
    main()
