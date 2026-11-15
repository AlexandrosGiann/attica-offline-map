"""Βήμα 1: αποκοπή bbox Περιφέρειας Αττικής από μεγάλο .osm.pbf (π.χ. όλη η Ευρώπη).

pass 1 (nodes): παράλληλο. Το PBF χωρίζεται σε τμήματα blobs. Κάθε worker παίρνει
          OSMHeader + τα blobs του μέσω stdin pipe και κρατά nodes μέσα στο bbox.
pass 2 (ways):  ένα process, φίλτρο C++ (IdTracker: όποια way αγγίζει node του bbox).
pass 3 (relations): ίδιο, όποια relation αναφέρει κρατημένο node/way.

Χρήση: python extract.py <input.osm.pbf> <output.osm.pbf> [workers]
"""
import os
import struct
import subprocess
import sys
import threading
import time

BBOX = (22.85, 35.80, 24.20, 38.42)  # minlon, minlat, maxlon, maxlat (Κύθηρα..Ωρωπός)
if os.environ.get("BBOX"):
    BBOX = tuple(float(v) for v in os.environ["BBOX"].split(","))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        if c < 0x80:
            return r, i
        s += 7


def blob_index(path):
    """[(offset, length, type)] για κάθε fileblock του PBF."""
    out = []
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        off = 0
        while off < size:
            f.seek(off)
            raw = f.read(4)
            if len(raw) < 4:
                break
            hl = struct.unpack(">I", raw)[0]
            hdr = f.read(hl)
            i, typ, ds = 0, b"", 0
            while i < len(hdr):  # BlobHeader: 1=type(str) 2=indexdata 3=datasize
                key, i = _varint(hdr, i)
                fn, wt = key >> 3, key & 7
                if wt == 2:
                    ln, i = _varint(hdr, i)
                    if fn == 1:
                        typ = hdr[i:i + ln]
                    i += ln
                elif wt == 0:
                    v, i = _varint(hdr, i)
                    if fn == 3:
                        ds = v
                else:
                    raise ValueError(f"μη αναμενόμενο wire type {wt} @ {off}")
            total = 4 + hl + ds
            if off + total > size:
                raise ValueError(f"κομμένο αρχείο @ {off}")
            out.append((off, total, typ))
            off += total
    return out


# ---------------- worker (pass 1) ----------------
def worker(dst):
    import osmium

    x0, y0, x1, y1 = BBOX
    w = osmium.SimpleWriter(dst)
    n = 0
    try:
        for o in osmium.FileProcessor(osmium.io.File("-", "pbf"), osmium.osm.NODE):
            loc = o.location
            if loc.valid():
                lo, la = loc.lon, loc.lat
                if x0 <= lo <= x1 and y0 <= la <= y1:
                    w.add_node(o)
                    n += 1
    finally:
        w.close()
    print(n, flush=True)


def feed(proc, path, header, ranges):
    try:
        with open(path, "rb") as f:
            f.seek(header[0])
            proc.stdin.write(f.read(header[1]))
            for off, ln in ranges:
                f.seek(off)
                rem = ln
                while rem:
                    chunk = f.read(min(rem, 1 << 24))
                    if not chunk:
                        raise IOError("απρόσμενο EOF")
                    proc.stdin.write(chunk)
                    rem -= len(chunk)
    except (BrokenPipeError, OSError):
        pass
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass


def main(src, dst, nworkers):
    import osmium

    if not os.path.isfile(src):
        sys.exit(f"Δεν βρέθηκε: {src}")
    tmpdir = dst + ".tmp"
    os.makedirs(tmpdir, exist_ok=True)

    log("index blobs")
    blobs = blob_index(src)
    header = [b for b in blobs if b[2] == b"OSMHeader"]
    if not header:
        sys.exit("χωρίς OSMHeader")
    header = header[0][:2]
    data = [(o, l) for o, l, t in blobs if t == b"OSMData"]
    log("blobs", len(data))

    # pass 1 -- τα node blobs είναι πρώτα, αλλά δεν ξέρουμε πού τελειώνουν: όλα τα
    # blobs σε συνεχόμενα τμήματα. Περισσότερα τμήματα από workers για ισοκατανομή.
    nchunks = nworkers * 4
    step = -(-len(data) // nchunks)
    chunks = [data[i:i + step] for i in range(0, len(data), step)]
    outs = [os.path.join(tmpdir, f"n{i:03d}.osm.pbf") for i in range(len(chunks))]
    todo = list(range(len(chunks)))
    running = []
    total = 0
    log("pass 1: nodes,", len(chunks), "chunks,", nworkers, "workers")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    while todo or running:
        while todo and len(running) < nworkers:
            i = todo.pop(0)
            if os.path.exists(outs[i]):
                os.remove(outs[i])
            p = subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "--worker", outs[i]],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
            )
            t = threading.Thread(target=feed, args=(p, src, header, chunks[i]), daemon=True)
            t.start()
            running.append((i, p, t))
        time.sleep(0.5)
        for r in running[:]:
            i, p, t = r
            if p.poll() is not None:
                t.join()
                out = p.stdout.read().decode().strip()
                if p.returncode != 0:
                    sys.exit(f"worker {i} απέτυχε ({p.returncode})")
                total += int(out or 0)
                running.remove(r)
                log(f"chunk {i + 1}/{len(chunks)} nodes={out} σύνολο={total}")

    tracker = osmium.IdTracker()
    tmp = dst + ".part.osm.pbf"
    if os.path.exists(tmp):
        os.remove(tmp)
    w = osmium.SimpleWriter(tmp)
    try:
        for o_ in outs:  # τα chunks είναι σε σειρά αρχείου -> ταξινομημένα ids
            for o in osmium.FileProcessor(o_, osmium.osm.NODE):
                tracker.add_node(o.id)
                w.add_node(o)
        log("pass 2: forward references (C++)")
        tracker.complete_forward_references(src, relation_depth=1)
        log("pass 3: ways + relations")
        n = {"w": 0, "r": 0}
        for o in osmium.FileProcessor(src, osmium.osm.WAY | osmium.osm.RELATION).with_filter(
            tracker.id_filter()
        ):
            if o.is_way():
                w.add_way(o)
                n["w"] += 1
            else:
                w.add_relation(o)
                n["r"] += 1
        log("ways", n["w"], "relations", n["r"])
    finally:
        w.close()
    os.replace(tmp, dst)
    for o_ in outs:
        os.remove(o_)
    os.rmdir(tmpdir)
    log("ok", dst, os.path.getsize(dst))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else os.cpu_count() or 4)
