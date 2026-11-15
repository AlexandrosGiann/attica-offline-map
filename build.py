"""Βήμα 3: χτίσιμο του συμπαγούς χάρτη attica.amap.
Χρήση: python build.py <attica_bbox.osm.pbf> <wv.json> <out.amap>

Περιέχει: γράφο δρόμων (κόμβοι = διασταυρώσεις, ακμές με γεωμετρία/μήκος/ταχύτητα/
πρόσβαση/όλα τα tags του δρόμου), όλα τα POI με όλα τα χρήσιμα tags, στάσεις,
γραμμές ΜΜΜ (λεωφορεία, τρόλεϊ, τραμ, μετρό, ηλεκτρικός, προαστιακός, πλοία) με σειρά
στάσεων και διαδρομή, όριο Περιφέρειας Αττικής, άρθρα + listings Wikivoyage.
"""
import json
import re
import sys
import time
from collections import Counter

import numpy as np
import osmium

import amap
from amap import SCALE, Polygon, haversine, morton

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# ---------------- κανόνες ----------------
JUNK = re.compile(r"^(source|created_by|note|fixme|FIXME|todo|check_date|survey|attribution|"
                  r"import|tiger|gnis|lacounty|osak|mapillary|old_ref_legislative|"
                  r"is_in|history|converted_by|geobase|naptan)(:|$)")
POI_KEYS = {"amenity", "shop", "tourism", "leisure", "historic", "office", "craft", "healthcare",
            "public_transport", "railway", "place", "emergency", "sport", "aeroway", "aerialway",
            "military", "man_made", "mountain_pass", "club", "information", "memorial",
            "attraction", "route_ref", "bus", "ferry", "water_point", "telecom", "education",
            "entrance", "embassy", "diplomatic", "government", "playground", "beach"}
NODE_ONLY_KEYS = {"highway", "natural", "barrier", "traffic_calming", "crossing", "ford", "waterway"}
NAME_PFX = ("name", "alt_name", "old_name", "int_name", "official_name", "short_name",
            "loc_name", "addr:", "wikipedia", "wikidata", "brand")
SKIP_VALUES = {("natural", "tree"), ("natural", "tree_row"), ("barrier", "fence"),
               ("barrier", "wall"), ("barrier", "kerb"), ("barrier", "retaining_wall"),
               ("highway", "street_lamp"), ("man_made", "utility_pole"), ("power", "pole"),
               ("power", "tower")}

ROAD_SPEED = {"motorway": 110, "motorway_link": 60, "trunk": 90, "trunk_link": 50,
              "primary": 70, "primary_link": 45, "secondary": 55, "secondary_link": 40,
              "tertiary": 45, "tertiary_link": 35, "unclassified": 35, "residential": 30,
              "living_street": 10, "service": 20, "road": 30, "track": 15, "busway": 40}
FOOTISH = ["footway", "path", "cycleway", "bridleway", "steps", "pedestrian", "corridor"]
HW = list(ROAD_SPEED) + FOOTISH + ["ferry"]
HW_ID = {h: i for i, h in enumerate(HW)}
TRANSIT = {"bus", "trolleybus", "tram", "subway", "train", "light_rail", "monorail", "ferry",
           "funicular", "share_taxi", "coach"}
DENY = {"no", "private", "agricultural", "forestry", "emergency", "military", "discouraged_no"}
F = dict(car_f=1, car_b=2, foot=4, bike_f=8, bike_b=16, ferry=32, toll=64, tunnel=128,
         bridge=256, lit=512, roundabout=1024, psv=2048)


def clean_tags(t):
    return {k: v for k, v in t.items() if not JUNK.match(k) and "\0" not in k and "\0" not in v}


def poi_worthy(t, is_node):
    for k, v in t.items():
        if (k, v) in SKIP_VALUES:
            continue
        if k in POI_KEYS or k.startswith(NAME_PFX) or (is_node and k in NODE_ONLY_KEYS):
            return True
    return False


def first(t, *keys):
    for k in keys:
        if k in t:
            return t[k]
    return None


def maxspeed(v):
    if not v:
        return 0
    v = v.strip().lower()
    table = {"gr:urban": 50, "gr:rural": 90, "gr:motorway": 130, "gr:trunk": 110,
             "walk": 5, "gr:living_street": 20}
    if v in table:
        return table[v]
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(mph)?", v)
    if not m:
        return 0
    s = float(m.group(1)) * (1.609 if m.group(2) else 1)
    return int(min(max(s, 1), 250))


def duration_s(v):
    if not v:
        return 0
    m = re.match(r"^(\d+):(\d\d)(?::(\d\d))?$", v.strip())
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3) or 0)
    m = re.match(r"^(\d+)$", v.strip())
    return int(m.group(1)) * 60 if m else 0


def road_attrs(t):
    """-> (κλάση, flags, ταχύτητα km/h) ή None αν δεν είναι δρόμος/πλοίο."""
    hw = t.get("highway")
    if t.get("route") == "ferry":
        f = F["ferry"] | F["foot"] | F["bike_f"] | F["bike_b"]
        if first(t, "motorcar", "motor_vehicle", "vehicle") not in ("no",):
            f |= F["car_f"] | F["car_b"]
        L = 0  # η ταχύτητα θα βγει από duration/μήκος
        return HW_ID["ferry"], f, L
    if hw not in HW_ID or hw == "ferry":
        return None
    if t.get("area") == "yes" and hw not in ("pedestrian", "footway"):
        return None
    f = 0
    car_ok = hw in ROAD_SPEED and hw != "busway"
    ca = first(t, "motorcar", "motor_vehicle", "vehicle", "access")
    if ca in DENY or ca in ("psv", "bus", "delivery"):
        car_ok = False
    if hw == "busway" or (ca in ("psv", "bus")):
        f |= F["psv"]
    fa = first(t, "foot", "access")
    foot_ok = hw not in ("motorway", "motorway_link")
    if t.get("foot") in ("yes", "designated", "permissive"):
        foot_ok = True
    elif fa in DENY:
        foot_ok = False
    ba = first(t, "bicycle", "vehicle", "access")
    bike_ok = hw not in ("motorway", "motorway_link", "steps", "footway", "pedestrian", "corridor")
    if t.get("bicycle") in ("yes", "designated", "permissive", "dismount"):
        bike_ok = True
    elif ba in DENY:
        bike_ok = False
    ow = t.get("oneway", "")
    fwd = bwd = True
    if ow in ("yes", "1", "true"):
        bwd = False
    elif ow in ("-1", "reverse"):
        fwd = False
    elif ow not in ("no", "false", "0") and (
            t.get("junction") in ("roundabout", "circular") or hw in ("motorway",)):
        bwd = False
    if car_ok:
        f |= (F["car_f"] if fwd else 0) | (F["car_b"] if bwd else 0)
    if foot_ok:
        f |= F["foot"]
    if bike_ok:
        bf, bb = fwd, bwd
        if t.get("oneway:bicycle") == "no" or str(t.get("cycleway", "")).startswith("opposite"):
            bf = bb = True
        if hw in FOOTISH:
            bf = bb = True
        f |= (F["bike_f"] if bf else 0) | (F["bike_b"] if bb else 0)
    for k, bit in (("toll", "toll"), ("lit", "lit")):
        if t.get(k) == "yes":
            f |= F[bit]
    if t.get("tunnel") not in (None, "no"):
        f |= F["tunnel"]
    if t.get("bridge") not in (None, "no"):
        f |= F["bridge"]
    if t.get("junction") in ("roundabout", "circular"):
        f |= F["roundabout"]
    sp = maxspeed(t.get("maxspeed")) or ROAD_SPEED.get(hw, 5)
    return HW_ID[hw], f, sp


# ---------------- ανάγνωση ----------------
def read_relations(path):
    rels = {}
    for o in osmium.FileProcessor(path, osmium.osm.RELATION):
        t = {x.k: x.v for x in o.tags}
        typ = t.get("type")
        keep = (typ in ("route", "route_master") and t.get("route", t.get("route_master")) in TRANSIT) \
            or (t.get("public_transport") == "stop_area")
        if keep:
            rels[o.id] = (clean_tags(t), [(m.type, m.ref, m.role) for m in o.members])
        if t.get("boundary") == "administrative":
            nm = " ".join(t.get(k, "") for k in ("name", "name:el", "name:en", "official_name"))
            if ("Αττικ" in nm or "Attic" in nm) and t.get("admin_level") in ("4", "5"):
                bnd[o.id] = (clean_tags(t), [(m.type, m.ref, m.role) for m in o.members])
    return rels


bnd = {}


def stitch_rings(wayrefs):
    """wayrefs: [(ids, lon, lat)] -> κλειστά δαχτυλίδια με ένωση άκρων (κατά node id).
    Κόμβοι χωρίς θέση (έξω από το απόσπασμα) παραλείπονται από τις συντεταγμένες."""
    segs = [(list(i), list(x), list(y)) for i, x, y in wayrefs if len(i) >= 2]
    rings, open_ = [], 0
    while segs:
        ids, xs, ys = segs.pop()
        changed = True
        while ids[0] != ids[-1] and changed:
            changed = False
            for k, (i2, x2, y2) in enumerate(segs):
                if i2[0] == ids[-1]:
                    ids += i2[1:]; xs += x2[1:]; ys += y2[1:]
                elif i2[-1] == ids[-1]:
                    ids += i2[-2::-1]; xs += x2[-2::-1]; ys += y2[-2::-1]
                elif i2[-1] == ids[0]:
                    ids[:0] = i2[:-1]; xs[:0] = x2[:-1]; ys[:0] = y2[:-1]
                elif i2[0] == ids[0]:
                    ids[:0] = i2[:0:-1]; xs[:0] = x2[:0:-1]; ys[:0] = y2[:0:-1]
                else:
                    continue
                segs.pop(k)
                changed = True
                break
        if ids[0] != ids[-1]:
            open_ += 1
        pts = np.column_stack([xs, ys])
        pts = pts[~np.isnan(pts).any(1)]
        if len(pts) >= 3:
            rings.append(pts)
    return rings, open_


def read_main(path, route_way_ids, bnd_way_ids=frozenset()):
    pois = []          # (osmtype, id, lon, lat, tags, area_m2)
    roads = []         # (way_id, tags, ids, lon, lat)
    shapes = {}        # way_id -> (lon, lat) για γραμμές ΜΜΜ
    bshapes = {}       # way_id -> (ids, lon, lat) ορίων (nan αν λείπει θέση)
    fp = osmium.FileProcessor(path).with_locations().with_areas()
    for o in fp:
        if o.is_node():
            if len(o.tags) == 0:
                continue
            t = {x.k: x.v for x in o.tags}
            if poi_worthy(t, True) and o.location.valid():
                pois.append(("n", o.id, o.location.lon, o.location.lat, clean_tags(t), 0.0))
        elif o.is_way():
            t = {x.k: x.v for x in o.tags}
            if o.id in bnd_way_ids:
                bshapes[o.id] = ([n.ref for n in o.nodes],
                                 [n.location.lon if n.location.valid() else np.nan for n in o.nodes],
                                 [n.location.lat if n.location.valid() else np.nan for n in o.nodes])
            ra = road_attrs(t) if t else None
            want_shape = o.id in route_way_ids
            if ra is None and not want_shape:
                continue
            ids, lo, la = [], [], []
            for n in o.nodes:
                loc = n.location
                if loc.valid():
                    ids.append(n.ref)
                    lo.append(loc.lon)
                    la.append(loc.lat)
            if len(ids) < 2:
                continue
            if ra is not None:
                roads.append((o.id, clean_tags(t), np.array(ids, np.int64),
                              np.array(lo), np.array(la), ra))
            if want_shape:
                shapes[o.id] = (np.array(lo), np.array(la))
        elif o.is_area():
            t = {x.k: x.v for x in o.tags}
            if t.get("highway") and t.get("area") != "yes":
                continue
            if t.get("type") == "multipolygon" and len(t) == 1:
                continue
            if not poi_worthy(t, False):
                continue
            best, bA, bx, by = None, -1, 0, 0
            for r in o.outer_rings():
                pts = np.array([(n.lon, n.lat) for n in r if n.location.valid()])
                if len(pts) < 3:
                    continue
                x, y = pts[:, 0], pts[:, 1]
                cr = x[:-1] * y[1:] - x[1:] * y[:-1]
                A = cr.sum() / 2
                if abs(A) > bA:
                    if abs(A) < 1e-14:
                        cx, cy = x.mean(), y.mean()
                    else:
                        cx = ((x[:-1] + x[1:]) * cr).sum() / (6 * A)
                        cy = ((y[:-1] + y[1:]) * cr).sum() / (6 * A)
                    best, bA, bx, by = True, abs(A), cx, cy
            if best is None:
                continue
            m2 = bA * (111320 ** 2) * np.cos(np.radians(by))
            pois.append(("w" if o.from_way() else "r", o.orig_id(), bx, by, clean_tags(t), m2))
    return pois, roads, shapes, bshapes


# ---------------- strings ----------------
class Strings:
    def __init__(self):
        self.c = Counter()

    def count(self, tags):
        for k, v in tags.items():
            self.c[k] += 1
            self.c[v] += 1

    def freeze(self):
        self.lst = [s for s, _ in self.c.most_common()]
        self.id = {s: i for i, s in enumerate(self.lst)}

    def kv(self, tags_list):
        """-> (πλήθος ανά αντικείμενο, ζεύγη ids)"""
        cnt = np.fromiter((len(t) for t in tags_list), np.int64, len(tags_list))
        flat = [self.id[x] for t in tags_list for kv in sorted(t.items()) for x in kv]
        return cnt, np.array(flat, np.int64)


def q(v):
    return np.round(np.asarray(v) * SCALE).astype(np.int64)


# ---------------- γράφος ----------------
def build_graph(roads, poly):
    log("γράφος: δρόμοι", len(roads))
    lens = np.array([len(r[2]) for r in roads], np.int64)
    ids = np.concatenate([r[2] for r in roads])
    lon = np.concatenate([r[3] for r in roads])
    lat = np.concatenate([r[4] for r in roads])
    way = np.repeat(np.arange(len(roads)), lens)
    pos = np.arange(ids.size) - np.repeat(np.cumsum(lens) - lens, lens)
    ferry = np.array([r[5][0] == HW_ID["ferry"] for r in roads])[way]
    uniq, inv, cnt = np.unique(ids, return_inverse=True, return_counts=True)
    ulon = np.zeros(uniq.size)
    ulat = np.zeros(uniq.size)
    ulon[inv] = lon
    ulat[inv] = lat
    inside_u = poly.contains(ulon, ulat)
    inside = inside_u[inv] | ferry
    # ferry: κράτα ολόκληρο μόνο αν αγγίζει την Αττική
    fw = np.unique(way[ferry & inside_u[inv]])
    inside &= ~ferry | np.isin(way, fw)
    log("κόμβοι δρόμων", uniq.size, "μέσα", int(inside_u.sum()))
    K = np.flatnonzero(inside)
    kw = way[K]
    newrun = np.ones(K.size, bool)
    newrun[1:] = (kw[1:] != kw[:-1]) | (K[1:] != K[:-1] + 1)
    endrun = np.ones(K.size, bool)
    endrun[:-1] = newrun[1:]
    isv = (cnt[inv[K]] > 1) | newrun | endrun
    # κόμβος κορυφή αν είναι κορυφή σε οποιαδήποτε εμφάνιση
    vflag = np.zeros(uniq.size, bool)
    vflag[inv[K][isv]] = True
    isv = vflag[inv[K]]
    isv |= newrun | endrun
    run = np.cumsum(newrun) - 1
    # αθροιστική απόσταση ανά run
    d = np.zeros(K.size)
    d[1:] = haversine(lat[K[:-1]], lon[K[:-1]], lat[K[1:]], lon[K[1:]])
    d[newrun] = 0
    cum = np.cumsum(d)
    V = np.flatnonzero(isv)
    same = run[V[1:]] == run[V[:-1]]
    a, b = V[:-1][same], V[1:][same]
    # κορυφές με Morton σειρά
    vu = np.unique(inv[K][V])
    qx, qy = q(ulon[vu]), q(ulat[vu])
    order = np.argsort(morton(qx, qy), kind="stable")
    vu = vu[order]
    vid = np.full(uniq.size, -1, np.int64)
    vid[vu] = np.arange(vu.size)
    u = vid[inv[K[a]]]
    v = vid[inv[K[b]]]
    length = cum[b] - cum[a]
    keep = u != v
    a, b, u, v, length = a[keep], b[keep], u[keep], v[keep], length[keep]
    ew = kw[a]
    # ταξινόμηση ακμών: δρόμοι κατά Morton της 1ης κορυφής, μετά κατά θέση
    wkey = np.full(len(roads), np.iinfo(np.int64).max, np.int64)
    np.minimum.at(wkey, ew, u)
    used = np.unique(ew)
    worder = used[np.argsort(wkey[used], kind="stable")]
    wnew = np.full(len(roads), -1, np.int64)
    wnew[worder] = np.arange(worder.size)
    o = np.lexsort((pos[K[a]], wnew[ew]))
    a, b, u, v, length, ew = a[o], b[o], u[o], v[o], length[o], ew[o]
    # γεωμετρία ενδιάμεσων σημείων
    ninter = b - a - 1
    inter = np.repeat(a + 1, ninter) + (np.arange(ninter.sum()) - np.repeat(np.cumsum(ninter) - ninter, ninter))
    gx, gy = q(lon[K[inter]]), q(lat[K[inter]])
    vx, vy = q(ulon[vu]), q(ulat[vu])
    px = np.empty_like(gx)
    py = np.empty_like(gy)
    if gx.size:
        px[1:], py[1:] = gx[:-1], gy[:-1]
        starts = np.cumsum(ninter) - ninter
        hs = ninter > 0
        px[starts[hs]] = vx[u[hs]]
        py[starts[hs]] = vy[u[hs]]
    cls = np.array([roads[i][5][0] for i in range(len(roads))], np.int64)[ew]
    flags = np.array([roads[i][5][1] for i in range(len(roads))], np.int64)[ew]
    speed = np.array([roads[i][5][2] for i in range(len(roads))], np.int64)[ew]
    # πλοία: ταχύτητα από duration ή 30 km/h
    for i in np.unique(ew[cls == HW_ID["ferry"]]):
        m = ew == i
        tot = length[m].sum()
        dur = duration_s(roads[i][1].get("duration"))
        sp = (tot / dur * 3.6) if dur > 0 else 30
        speed[m] = int(min(max(round(sp), 5), 80))
    log("κορυφές", vu.size, "ακμές", u.size, "σημεία γεωμ.", gx.size)
    return dict(vx=vx, vy=vy, osm=uniq[vu], u=u, v=v, len_dm=np.round(length * 10).astype(np.int64),
                cls=cls, flags=flags, speed=speed, way=wnew[ew], ninter=ninter,
                gdx=gx - px, gdy=gy - py, ways=[roads[i] for i in worder])


# ---------------- ΜΜΜ ----------------
def stitch(seq, shapes):
    out = []
    parts = [shapes[w] for w in seq if w in shapes]
    for i, (x, y) in enumerate(parts):
        pts = np.column_stack([x, y])
        if out:
            end = out[-1][-1]
            if np.allclose(pts[-1], end) and not np.allclose(pts[0], end):
                pts = pts[::-1]
        elif len(parts) > 1:
            nx = np.column_stack(parts[1])
            if any(np.allclose(pts[0], nx[j]) for j in (0, -1)):
                pts = pts[::-1]
        if out and np.allclose(pts[0], out[-1][-1]):
            pts = pts[1:]
        out.append(pts)
    return np.vstack(out) if out else np.zeros((0, 2))


def main(pbf, wvjson, out):
    log("relations")
    rels = read_relations(pbf)
    route_ways = {ref for t, ms in rels.values() for (mt, ref, role) in ms
                  if mt == "w" and not role.startswith(("platform", "stop"))}
    log("relations ΜΜΜ", len(rels), "ways διαδρομών", len(route_ways))
    if not bnd:
        sys.exit("δεν βρέθηκε relation ορίου Αττικής")
    bid = max(bnd, key=lambda k: (bnd[k][0].get("name") == "Περιφέρεια Αττικής",
                                  bnd[k][0].get("admin_level") == "4"))
    btags, bmem = bnd[bid]
    bways = [ref for mt, ref, role in bmem if mt == "w" and role in ("outer", "inner", "")]
    pois, roads, shapes, bshapes = read_main(pbf, route_ways, frozenset(bways))
    log("POI", len(pois), "δρόμοι", len(roads), "ways ορίου", len(bshapes), "/", len(bways))
    rings, open_ = stitch_rings([bshapes[w] for w in bways if w in bshapes])
    log("όριο:", bid, btags.get("name"), btags.get("admin_level"), "δαχτυλίδια", len(rings),
        "ανοιχτά", open_)
    if not rings:
        sys.exit("άδειο όριο")
    poly = Polygon(rings)

    g = build_graph(roads, poly)

    # POI μέσα στο όριο
    P = [p for p in pois]
    plon = np.array([p[2] for p in P])
    plat = np.array([p[3] for p in P])
    pin = poly.contains(plon, plat)
    P = [p for p, k in zip(P, pin) if k]
    px, py = q([p[2] for p in P]), q([p[3] for p in P])
    o = np.argsort(morton(px, py), kind="stable")
    P = [P[i] for i in o]
    px, py = px[o], py[o]
    pkey = {(p[0], p[1]): i for i, p in enumerate(P)}
    log("POI μέσα", len(P))

    # γραμμές
    routes, masters, areas = [], [], []
    shape_n, shape_x, shape_y = [], [], []
    for rid, (t, ms) in sorted(rels.items()):
        if t.get("type") == "route":
            stops = []
            for mt, ref, role in ms:
                if role.startswith(("stop", "platform")) and (mt, ref) in pkey:
                    k = pkey[(mt, ref)]
                    stops.append([k, role])
            shp = stitch([ref for mt, ref, role in ms if mt == "w" and not role.startswith(("platform", "stop"))], shapes)
            if shp.size:
                sp_in = poly.contains(shp[:, 0], shp[:, 1])
            if not stops and (not shp.size or not sp_in.any()):
                continue
            routes.append({"id": rid, "tags": t, "stops": stops})
            sx, sy = q(shp[:, 0]), q(shp[:, 1])
            shape_n.append(len(sx))
            shape_x.append(np.diff(sx, prepend=0))
            shape_y.append(np.diff(sy, prepend=0))
        elif t.get("type") == "route_master":
            masters.append({"id": rid, "tags": t, "routes": [ref for mt, ref, _ in ms if mt == "r"]})
        else:
            mem = [[pkey[(mt, ref)], role] for mt, ref, role in ms if (mt, ref) in pkey]
            if mem:
                areas.append({"id": rid, "tags": t, "members": mem})
    rids = {r["id"] for r in routes}
    masters = [m for m in masters if any(x in rids for x in m["routes"])]
    log("γραμμές", len(routes), "masters", len(masters), "stop_areas", len(areas))

    # Wikivoyage
    wv = json.load(open(wvjson, encoding="utf-8"))
    L = wv["listings"]
    lin = poly.contains(np.array([x["lon"] for x in L]), np.array([x["lat"] for x in L])) if L else []
    L = [x for x, k in zip(L, lin) if k]
    A = wv["articles"]
    ain = poly.contains(np.array([a["lon"] for a in A]), np.array([a["lat"] for a in A])) if A else []
    src = Counter(x["src"] for x in L)
    A = [a for a, k in zip(A, ain) if k or src[a["title"]] >= 3]
    log("Wikivoyage άρθρα", len(A), [a["title"] for a in A], "listings", len(L))

    # strings
    S = Strings()
    for p in P:
        S.count(p[4])
    for r in g["ways"]:
        S.count(r[1])
    S.freeze()

    w = amap.Writer()
    w.json("meta", {
        "format": 1, "name": btags.get("name"), "boundary_relation": bid,
        "bbox": [float(x) for x in (poly.x0, poly.y0, poly.x0 + poly.nx * poly.res, poly.y0 + poly.ny * poly.res)],
        "scale": SCALE, "highway_classes": HW, "flags": F,
        "sources": ["OpenStreetMap © OpenStreetMap contributors, ODbL",
                    "Wikivoyage (en), CC BY-SA"],
        "built": time.strftime("%Y-%m-%d"),
        "counts": {"vertices": int(g["vx"].size), "edges": int(g["u"].size), "pois": len(P),
                   "routes": len(routes), "articles": len(A), "listings": len(L)},
    })
    w.strings("str", S.lst)
    # όριο
    w.uints("bnd.n", [len(r) for r in rings])
    bx = np.concatenate([q(r[:, 0]) for r in rings])
    by = np.concatenate([q(r[:, 1]) for r in rings])
    w.ints("bnd.dx", np.diff(bx, prepend=0))
    w.ints("bnd.dy", np.diff(by, prepend=0))
    # γράφος
    w.ints("v.dx", np.diff(g["vx"], prepend=0))
    w.ints("v.dy", np.diff(g["vy"], prepend=0))
    w.ints("v.osm", np.diff(g["osm"], prepend=0))
    u, v = g["u"], g["v"]
    prev = np.concatenate([[0], v[:-1]])
    w.ints("e.u", u - prev)
    w.ints("e.v", v - u)
    w.uints("e.len", g["len_dm"])
    w.uints("e.cls", g["cls"])
    w.uints("e.flags", g["flags"])
    w.uints("e.speed", g["speed"])
    w.ints("e.way", np.diff(g["way"], prepend=0))
    w.uints("e.ni", g["ninter"])
    w.ints("g.dx", g["gdx"])
    w.ints("g.dy", g["gdy"])
    wc, wkv = S.kv([r[1] for r in g["ways"]])
    w.ints("way.osm", np.diff(np.array([r[0] for r in g["ways"]], np.int64), prepend=0))
    w.uints("way.nt", wc)
    w.uints("way.kv", wkv)
    # POI
    w.ints("p.dx", np.diff(px, prepend=0))
    w.ints("p.dy", np.diff(py, prepend=0))
    w.uints("p.osm", [(p[1] << 2) | {"n": 0, "w": 1, "r": 2}[p[0]] for p in P])
    w.uints("p.area", [int(round(p[5])) for p in P])
    pc, pkv = S.kv([p[4] for p in P])
    w.uints("p.nt", pc)
    w.uints("p.kv", pkv)
    # ΜΜΜ
    w.json("transit", {"routes": routes, "masters": masters, "stop_areas": areas})
    w.uints("rs.n", shape_n)
    w.ints("rs.dx", np.concatenate(shape_x) if shape_x else [])
    w.ints("rs.dy", np.concatenate(shape_y) if shape_y else [])
    # Wikivoyage
    w.json("wv", {"articles": A, "listings": L})
    raw, comp = w.save(out)
    log(f"ok {out}: {comp / 1e6:.2f} MB (ασυμπίεστο {raw / 1e6:.2f} MB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
