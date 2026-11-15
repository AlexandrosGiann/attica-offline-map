"""Demo βίντεο του χάρτη Αττικής: γράφος, ΜΜΜ, Dijkstra (αυτοκίνητο/ΜΜΜ/πλοίο).
Χρήση: python demo_video.py [out.mp4]
"""
import heapq
import math
import sys
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from attica import Map, fmt_d, fmt_t

W, H, FPS = 1280, 720, 30
BAR = 96
BG = (14, 17, 23)
FONT = "C:/Windows/Fonts/segoeui.ttf"
FONTB = "C:/Windows/Fonts/segoeuib.ttf"


def font(sz, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, sz)


CLS_STYLE = {  # χρώμα, πάχος, σειρά
    "motorway": ((255, 140, 60), 3, 9), "motorway_link": ((255, 140, 60), 2, 8),
    "trunk": ((250, 190, 70), 3, 8), "trunk_link": ((250, 190, 70), 2, 7),
    "primary": ((235, 210, 110), 2, 7), "primary_link": ((235, 210, 110), 1, 6),
    "secondary": ((200, 200, 170), 2, 6), "secondary_link": ((200, 200, 170), 1, 5),
    "tertiary": ((150, 155, 160), 1, 5), "tertiary_link": ((150, 155, 160), 1, 4),
    "ferry": ((80, 150, 255), 1, 9),
}
DEFAULT = ((70, 76, 86), 1, 1)
FOOT = ((48, 52, 60), 1, 0)
LINE_COL = {"subway": (60, 200, 120), "train": (230, 80, 200), "light_rail": (60, 200, 120),
            "tram": (90, 220, 230), "bus": (230, 120, 60), "trolleybus": (240, 200, 60),
            "ferry": (80, 150, 255), "funicular": (200, 200, 200)}


class View:
    def __init__(self, lon0, lat0, lon1, lat1):
        c = math.cos(math.radians((lat0 + lat1) / 2))
        mw, mh = W - 40, H - BAR - 30
        s = min(mw / ((lon1 - lon0) * c), mh / (lat1 - lat0))
        self.s, self.c = s, c
        self.cx, self.cy = (lon0 + lon1) / 2, (lat0 + lat1) / 2
        self.ox, self.oy = W / 2, (H - BAR) / 2 + 5

    def xy(self, lon, lat):
        x = self.ox + (np.asarray(lon) - self.cx) * self.c * self.s
        y = self.oy - (np.asarray(lat) - self.cy) * self.s
        return x, y


def render_roads(M, view, detail=True):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    names = M.HW
    style = [CLS_STYLE.get(h, FOOT if h in ("footway", "path", "cycleway", "bridleway", "steps",
                                             "pedestrian", "corridor") else DEFAULT) for h in names]
    order = np.argsort([style[c][2] for c in M.ecls], kind="stable")
    ex = np.concatenate([[0], np.cumsum(M.eni)])
    ux, uy = view.xy(M.vlon, M.vlat)
    gx, gy = view.xy(M.glon, M.glat)
    for e in order.tolist():
        col, wd, rank = style[M.ecls[e]]
        if not detail and rank < 4:
            continue
        a, b = ex[e], ex[e + 1]
        u, v = M.eu[e], M.ev[e]
        pts = [(ux[u], uy[u])] + list(zip(gx[a:b], gy[a:b])) + [(ux[v], uy[v])]
        xs = [p[0] for p in pts]
        if max(xs) < -50 or min(xs) > W + 50:
            continue
        d.line(pts, fill=col, width=wd)
    return img


def draw_boundary(img, M, view, col=(120, 130, 150)):
    d = ImageDraw.Draw(img)
    for r in M.boundary:
        x, y = view.xy(r[:, 0], r[:, 1])
        d.line(list(zip(x, y)), fill=col, width=1)


def caption(img, title, sub="", right=""):
    d = ImageDraw.Draw(img)
    d.rectangle([0, H - BAR, W, H], fill=(22, 26, 34))
    d.text((24, H - BAR + 12), title, font=font(30, True), fill=(240, 240, 245))
    if sub:
        d.text((24, H - BAR + 54), sub, font=font(20), fill=(170, 180, 195))
    if right:
        d.text((W - 24, H - BAR + 20), right, font=font(20), fill=(140, 200, 255), anchor="ra")
    return img


def marker(d, x, y, col, label=None):
    d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=col, outline=(255, 255, 255), width=2)
    if label:
        f = font(18, True)
        d.text((x + 11, y - 12), label, font=f, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))


def record(M, g, sources, targets):
    """Ίδιος Dijkstra με το attica.py, αλλά κρατά τη σειρά οριστικοποίησης κορυφών."""
    ptr, dst, w = g["ptr"], g["dst_l"], g["w_l"]
    dist, h, done, order = {}, [], set(), []
    best = math.inf
    for v, c in sources.items():
        dist[v] = c
        heapq.heappush(h, (c, v))
    while h:
        c, v = heapq.heappop(h)
        if v in done:
            continue
        if c >= best:
            break
        done.add(v)
        order.append(v)
        if v in targets:
            best = min(best, c + targets[v])
        for i in range(ptr[v], ptr[v + 1]):
            x, nc = dst[i], c + w[i]
            if nc < dist.get(x, math.inf):
                dist[x] = nc
                heapq.heappush(h, (nc, x))
    return order


class Writer:
    def __init__(self, path):
        self.w = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8,
                                    macro_block_size=16, pixelformat="yuv420p")
        self.n = 0

    def add(self, img, frames=1):
        a = np.asarray(img.convert("RGB"))
        for _ in range(frames):
            self.w.append_data(a)
            self.n += 1

    def fade(self, a, b, frames=15):
        for i in range(1, frames + 1):
            self.add(Image.blend(a, b, i / frames))

    def close(self):
        self.w.close()


def draw_path(img, view, lon, lat, col, width=5):
    d = ImageDraw.Draw(img)
    x, y = view.xy(lon, lat)
    pts = list(zip(x, y))
    if len(pts) > 1:
        d.line(pts, fill=(0, 0, 0), width=width + 4)
        d.line(pts, fill=col, width=width)


def main(out):
    t0 = time.time()
    M = Map()
    c = M.meta["counts"]
    vid = Writer(out)
    blank = Image.new("RGB", (W, H), BG)

    # 1. Ολόκληρη η Αττική
    lo = np.concatenate([r[:, 0] for r in M.boundary]); la = np.concatenate([r[:, 1] for r in M.boundary])
    vall = View(lo.min(), la.min(), lo.max(), la.max())
    full = render_roads(M, vall, detail=False)
    draw_boundary(full, M, vall)
    s1 = caption(full.copy(), "Χάρτης Αττικής ως γράφος + Dijkstra",
                 f"{c['vertices']:,} κόμβοι · {c['edges']:,} ακμές · {c['pois']:,} σημεία ενδιαφέροντος · "
                 f"{c['routes']} γραμμές ΜΜΜ".replace(",", "."),
                 f"attica.amap · {M.file_size / 1e6:.2f} MB")
    vid.fade(caption(blank.copy(), "Χάρτης Αττικής ως γράφος + Dijkstra"), s1, 20)
    vid.add(s1, FPS * 3)
    print("σκηνή 1", round(time.time() - t0), "s", flush=True)

    # 2. Λεκανοπέδιο: δρόμοι + γραμμές ΜΜΜ + στάσεις
    vath = View(23.58, 37.86, 23.92, 38.10)
    ath = render_roads(M, vath)
    s2a = caption(ath.copy(), "Οδικό δίκτυο λεκανοπεδίου",
                  "Κάθε δρόμος: μήκος, ταχύτητα, μονόδρομος, πρόσβαση αυτοκίνητο / πεζός / ποδήλατο")
    vid.fade(s1, s2a, 20)
    vid.add(s2a, FPS * 2)
    tr = ath.copy()
    d = ImageDraw.Draw(tr)
    for kind in ("bus", "trolleybus", "tram", "train", "subway", "light_rail", "ferry"):
        for r in M.routes:
            if r["tags"].get("route") != kind or r["shape"][0].size < 2:
                continue
            x, y = vath.xy(*r["shape"])
            col = LINE_COL.get(kind, (200, 200, 200))
            if kind == "subway":
                col = {"M1": (60, 200, 120), "M2": (230, 60, 60), "M3": (60, 120, 230)}.get(
                    r["tags"].get("ref", "").replace("Μ", "M"), col)
            d.line(list(zip(x, y)), fill=col, width=4 if kind in ("subway", "train", "tram") else 1)
    stops = [i for i in range(M.plon.size) if M._is_stop_tags(i)]
    sx, sy = vath.xy(M.plon[stops], M.plat[stops])
    for x, y in zip(sx, sy):
        if 0 <= x < W and 0 <= y < H - BAR:
            d.rectangle([x - 1, y - 1, x + 1, y + 1], fill=(255, 255, 255))
    s2b = caption(tr, "Στάσεις και γραμμές ΜΜΜ",
                  f"{len(stops):,} στάσεις · λεωφορεία, τρόλεϊ, μετρό Μ1/Μ2/Μ3, τραμ, Προαστιακός, πλοία".replace(",", "."))
    vid.fade(s2a, s2b, 20)
    vid.add(s2b, int(FPS * 3.5))
    print("σκηνή 2", round(time.time() - t0), "s", flush=True)

    # 3. Dijkstra με αυτοκίνητο: εξερεύνηση + διαδρομή
    A, B = M.resolve("Κηφισιά"), M.resolve("Πειραιάς")
    g = M.graph("car")
    sa, _ = M.snap(A["lat"], A["lon"], g)
    sb, _ = M.snap(B["lat"], B["lon"], g)
    order = record(M, g, M._ends(g, sa, False), M._ends(g, sb, True))
    res = M.route(A, B, "car")
    base = ath.copy()
    dd = ImageDraw.Draw(base)
    ax, ay = vath.xy(A["lon"], A["lat"]); bx, by = vath.xy(B["lon"], B["lat"])
    marker(dd, ax, ay, (60, 220, 120), A["name"]); marker(dd, bx, by, (240, 80, 80), B["name"])
    arr = np.asarray(base).copy()
    ox, oy = vath.xy(M.vlon[order], M.vlat[order])
    ox, oy = ox.astype(int), oy.astype(int)
    ok = (ox >= 1) & (ox < W - 1) & (oy >= 1) & (oy < H - BAR - 1)
    nf = FPS * 5
    sub = "Ο Dijkstra οριστικοποιεί κορυφές με σειρά χρόνου από την αφετηρία (binary heap)"
    for k in range(nf):
        a, b = len(order) * k // nf, len(order) * (k + 1) // nf
        m = ok[a:b]
        xs, ys = ox[a:b][m], oy[a:b][m]
        for dx in (0, 1):
            for dy in (0, 1):
                arr[ys + dy, xs + dx] = (255, 170, 40)
        fr = Image.fromarray(arr)
        marker(ImageDraw.Draw(fr), ax, ay, (60, 220, 120), A["name"])
        marker(ImageDraw.Draw(fr), bx, by, (240, 80, 80), B["name"])
        vid.add(caption(fr, f"Dijkstra · αυτοκίνητο · {b:,} κορυφές".replace(",", "."), sub))
    fin = Image.fromarray(arr)
    draw_path(fin, vath, res["lon"], res["lat"], (80, 220, 255), 5)
    marker(ImageDraw.Draw(fin), ax, ay, (60, 220, 120), A["name"])
    marker(ImageDraw.Draw(fin), bx, by, (240, 80, 80), B["name"])
    s3 = caption(fin, f"{A['name']} → {B['name']} με αυτοκίνητο: ~{fmt_t(res['cost'])}, {fmt_d(res['dist'])}",
                 f"{res['settled']:,} κορυφές σε {res['dijkstra_s'] * 1000:.0f} ms".replace(",", "."),
                 "συντομότερη διαδρομή")
    vid.add(s3, int(FPS * 3.5))
    print("σκηνή 3", round(time.time() - t0), "s", flush=True)

    # 4. ΜΜΜ: Κηφισιά -> Αεροδρόμιο
    A2, B2 = M.resolve("Σύνταγμα"), M.resolve("Αεροδρόμιο Αθηνών")
    r2 = M.route(A2, B2, "transit")
    vt = View(23.70, 37.90, 23.99, 38.08)
    bg4 = render_roads(M, vt)
    draw_path(bg4, vt, r2["lon"], r2["lat"], (230, 80, 200), 5)
    d4 = ImageDraw.Draw(bg4)
    yy = 18
    for s in r2["steps"]:
        if s["type"] == "ride":
            p, q = s["from"], s["to"]
            for pp, col in ((p, (60, 220, 120)), (q, (240, 80, 80))):
                x, y = vt.xy(M.plon[pp], M.plat[pp])
                marker(d4, x, y, col, M.poi_name(pp))
            txt = f"{s['name'][:60]} · {s['stops']} στάσεις · ~{fmt_t(s['time'])}"
            d4.rounded_rectangle([16, yy - 4, 26 + d4.textlength(txt, font=font(18)), yy + 26],
                                 6, fill=(22, 26, 34))
            d4.text((22, yy), txt, font=font(18), fill=(235, 235, 240))
            yy += 36
    s4 = caption(bg4, f"ΜΜΜ: {A2['name']} → {B2['name']} ~{fmt_t(r2['cost'])}",
                 "Πολυτροπικός γράφος: περπάτημα + επιβίβαση (αναμονή) + διαδρομή στάση-στάση")
    vid.fade(s3, s4, 20)
    vid.add(s4, FPS * 4)
    print("σκηνή 4", round(time.time() - t0), "s", flush=True)

    # 5. Πλοίο: Πειραιάς -> Αίγινα
    A3, B3 = M.resolve("Πειραιάς"), M.resolve("Αίγινα")
    r3 = M.route(A3, B3, "foot")
    vs = View(23.38, 37.68, 23.70, 37.97)
    bg5 = render_roads(M, vs)
    draw_boundary(bg5, M, vs)
    draw_path(bg5, vs, r3["lon"], r3["lat"], (80, 150, 255), 5)
    d5 = ImageDraw.Draw(bg5)
    for P_, col in ((A3, (60, 220, 120)), (B3, (240, 80, 80))):
        x, y = vs.xy(P_["lon"], P_["lat"])
        marker(d5, x, y, col, P_["name"])
    s5 = caption(bg5, f"Πεζός + πλοίο: {A3['name']} → {B3['name']} ~{fmt_t(r3['cost'])}",
                 "Οι γραμμές πλοίων είναι ακμές του γράφου (με χρόνο αναμονής)")
    vid.fade(s4, s5, 20)
    vid.add(s5, FPS * 3)

    # 6. Τέλος
    end = caption(full.copy(), "Ένα αρχείο, όλη η Αττική",
                  f"{M.file_size / 1e6:.2f} MB · OSM + Wikivoyage ({c['articles']} άρθρα) · μόνο numpy · "
                  "Dijkstra 1–400 ms", "© OpenStreetMap contributors · Wikivoyage")
    vid.fade(s5, end, 20)
    vid.add(end, FPS * 3)
    vid.close()
    print("ok", out, vid.n, "frames", round(vid.n / FPS, 1), "s, χρόνος", round(time.time() - t0), "s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "attica_demo.mp4")
