"""Βήμα 2: εξαγωγή άρθρων/listings Wikivoyage (ZIM) μέσα στο bbox της Αττικής.
Χρήση: python wikivoyage.py <wikivoyage.zim> <out.json>
Κρατά: άρθρα με geo.position στο bbox (πλήρες κείμενο ανά ενότητα) και κάθε listing
(See/Do/Eat/Sleep/...) με συντεταγμένες στο bbox, από οποιοδήποτε άρθρο.
"""
import html
import json
import re
import sys
import time
from html.parser import HTMLParser

from extract import BBOX

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
        "source", "track", "wbr"}
SKIP = {"style", "script", "noscript", "head", "title"}
BLOCK = {"p", "div", "li", "tr", "section", "dd", "dt", "ul", "ol", "table", "br",
         "blockquote", "figure", "figcaption", "h1", "h2", "h3", "h4", "h5", "h6"}
FIELDS = {"listing-name": "name", "listing-address": "address", "listing-directions":
          "directions", "listing-phone": "phone", "listing-email": "email",
          "listing-hours": "hours", "listing-price": "price", "listing-content": "desc",
          "listing-alt": "alt", "listing-checkin": "checkin", "listing-checkout": "checkout",
          "listing-tollfree": "tollfree", "listing-fax": "fax"}
GEO_META = re.compile(r'<meta name="geo.position" content="(-?[\d.]+);(-?[\d.]+)"')
WS = re.compile(r"[ \t ]+")


def inbox(lat, lon):
    x0, y0, x1, y1 = BBOX
    return x0 <= lon <= x1 and y0 <= lat <= y1


def clean(s):
    s = WS.sub(" ", s)
    s = re.sub(r" *\n[ \n]*", "\n", s)
    return s.strip()


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []          # (tag, flags)
        self.skip = 0
        self.h2 = ""
        self.h3 = ""
        self.head = None         # buffer επικεφαλίδας
        self.sections = []       # [heading, [text parts]]
        self.cur = ["", []]
        self.listings = []
        self.lst = None          # τρέχον listing
        self.field = []          # στοίβα ενεργών πεδίων listing

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = (a.get("class") or "").split()
        if tag in VOID:
            if tag == "br":
                self._text("\n")
            return
        flags = set()
        if tag in SKIP or "listing-metadata" in cls or "mw-kartographer-maplink" in cls \
                or "mw-editsection" in cls or "reference" in cls:
            self.skip += 1
            flags.add("skip")
        if tag in ("h2", "h3"):
            self.head = []
            flags.add("head")
        if "vcard" in cls and self.lst is None:
            self.lst = {"sec": self.h2, "sub": self.h3, "t": []}
            flags.add("vcard")
            if a.get("data-lat"):
                self._ll(a)
        if self.lst is not None:
            if a.get("data-lat") and "lat" not in self.lst:
                self._ll(a)
            for c in cls:
                if c in FIELDS:
                    flags.add("f:" + FIELDS[c])
                    self.field.append([FIELDS[c], []])
            if tag == "a" and a.get("href", "").startswith("http") and "url" not in self.lst:
                self.lst["url"] = a["href"]
            if "wikidata" in a.get("data-wikidata", "") or a.get("data-wikidata"):
                self.lst["wd"] = a["data-wikidata"]
        if tag in BLOCK:
            self._text("\n")
        self.stack.append((tag, flags))

    def _ll(self, a):
        try:
            self.lst["lat"] = float(a["data-lat"])
            self.lst["lon"] = float(a["data-lon"])
        except (KeyError, ValueError):
            pass

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        # βρες το αντίστοιχο άνοιγμα (ανοχή σε κακό HTML)
        for k in range(len(self.stack) - 1, -1, -1):
            if self.stack[k][0] == tag:
                break
        else:
            return
        while len(self.stack) > k:
            t, flags = self.stack.pop()
            self._close(t, flags)

    def _close(self, tag, flags):
        if "skip" in flags:
            self.skip -= 1
        for f in [x for x in flags if x.startswith("f:")]:
            name, parts = self.field.pop()
            v = clean("".join(parts))
            if v and self.lst is not None:
                self.lst.setdefault(name, v)
        if "head" in flags and self.head is not None:
            h = clean("".join(self.head))
            self.head = None
            if tag == "h2":
                self.h2, self.h3 = h, ""
            else:
                self.h3 = h
            self.sections.append(self.cur)
            self.cur = [("## " if tag == "h2" else "### ") + h, []]
        if "vcard" in flags and self.lst is not None:
            self.lst["text"] = clean("".join(self.lst.pop("t")))
            self.listings.append(self.lst)
            self.lst = None
        if tag in BLOCK:
            self._text("\n")

    def _text(self, s):
        if self.skip:
            return
        if self.head is not None:
            self.head.append(s)
            return
        self.cur[1].append(s)
        if self.lst is not None:
            self.lst["t"].append(s)
            for f in self.field:
                f[1].append(s)

    def handle_data(self, data):
        self._text(data)

    def result(self):
        self.sections.append(self.cur)
        secs = []
        for h, parts in self.sections:
            t = clean("".join(parts))
            if t or h.startswith("## "):
                secs.append([h, t])
        return secs


def main(zim, out):
    from libzim.reader import Archive

    a = Archive(zim)
    arts, lists = [], []
    t0 = time.time()
    n = a.all_entry_count
    for i in range(n):
        e = a.get_entry_by_id(i)
        if e.is_redirect:
            continue
        try:
            it = e.get_item()
        except Exception:
            continue
        if not it.mimetype.startswith("text/html"):
            continue
        raw = bytes(it.content)
        if b"data-lat" not in raw and b"geo.position" not in raw:
            continue
        h = raw.decode("utf-8", "replace")
        m = GEO_META.search(h)
        page_in = bool(m) and inbox(float(m.group(1)), float(m.group(2)))
        if not page_in and not any(
            inbox(float(la), float(lo))
            for la, lo in re.findall(r'data-lat="(-?[\d.]+)" data-lon="(-?[\d.]+)"', h)
        ):
            continue
        p = Page()
        try:
            p.feed(h)
            p.close()
        except Exception as ex:  # κακό HTML: συνέχισε
            print("parse error", e.title, ex, file=sys.stderr)
        for L in p.listings:
            if "lat" in L and inbox(L["lat"], L["lon"]):
                L["src"] = e.title
                lists.append(L)
        if page_in:
            arts.append({"title": e.title, "lat": float(m.group(1)), "lon": float(m.group(2)),
                         "sections": p.result()})
        if i % 20000 == 0:
            print(f"{i}/{n} {time.time() - t0:.0f}s άρθρα={len(arts)} listings={len(lists)}",
                  flush=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"articles": arts, "listings": lists}, f, ensure_ascii=False)
    print("ok άρθρα", len(arts), "listings", len(lists))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
