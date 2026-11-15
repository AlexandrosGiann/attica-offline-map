/* Χάρτης Αττικής — JavaScript εκδοχή του attica.py (γράφος + Dijkstra), χωρίς βιβλιοθήκες.
 * Διαβάζει το attica.bin (ίδια μορφή ενοτήτων με το .amap, χωρίς xz· ο server το στέλνει gzip).
 * Σύνταξη ES2017 ώστε να τρέχει και σε παλιούς Android browsers.
 */
(function () {
  "use strict";

  var WALK = 5 / 3.6, BIKE = 15 / 3.6, FERRY_WAIT = 1800, ACCESS_MAX = 500;
  var TRANSIT_PARAMS = { bus: [18, 600], trolleybus: [16, 600], coach: [45, 1800], tram: [18, 420],
    subway: [33, 240], light_rail: [30, 300], monorail: [30, 300], train: [55, 900], ferry: [30, 1800],
    funicular: [10, 600], share_taxi: [25, 600] };
  var TYPE_EL = { bus: "Λεωφορείο", trolleybus: "Τρόλεϊ", coach: "Υπεραστικό", tram: "Τραμ", subway: "Μετρό",
    light_rail: "Ηλεκτρικός", monorail: "Monorail", train: "Τρένο/Προαστιακός", ferry: "Πλοίο",
    funicular: "Τελεφερίκ", share_taxi: "Ταξί" };
  var TRANSIT_EDGE = -2147483648;
  var R = 6371008.8, D2R = Math.PI / 180;

  // ---------------- βοηθητικά ----------------
  function hav(la1, lo1, la2, lo2) {
    var a = Math.sin((la2 - la1) * D2R / 2), b = Math.sin((lo2 - lo1) * D2R / 2);
    var h = a * a + Math.cos(la1 * D2R) * Math.cos(la2 * D2R) * b * b;
    return 2 * R * Math.asin(Math.sqrt(Math.min(h, 1)));
  }
  function norm(s) {
    s = String(s).toLowerCase().replace(/ς/g, "σ");
    if (s.normalize) s = s.normalize("NFD").replace(/[̀-ͯ]/g, "");
    return s.replace(/[^0-9a-zͰ-Ͽἀ-῿]+/g, " ").trim();
  }
  function fmtT(s) {
    s = Math.round(s);
    var m = Math.floor(s / 60), h = Math.floor(m / 60);
    if (h) return h + "ω " + (m % 60) + "λ";
    if (s >= 30) return Math.max(m, 1) + "λ";
    return s + "δ";
  }
  function fmtD(m) { return m >= 1000 ? (m / 1000).toFixed(1) + " km" : Math.round(m) + " m"; }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }

  // ---------------- αποκωδικοποίηση ----------------
  function uvarints(b, off, len) {
    var end = off + len, n = 0, i;
    for (i = off; i < end; i++) if (b[i] < 128) n++;
    var out = new Float64Array(n), k = 0, v = 0, mul = 1;
    for (i = off; i < end; i++) {
      var c = b[i];
      v += (c & 127) * mul;
      if (c < 128) { out[k++] = v; v = 0; mul = 1; } else mul *= 128;
    }
    return out;
  }
  function zigzag(a) {
    for (var i = 0; i < a.length; i++) { var v = a[i]; a[i] = (v % 2) ? -(v + 1) / 2 : v / 2; }
    return a;
  }
  function parse(buf) {
    var b = new Uint8Array(buf), dv = new DataView(buf), i = 0, out = {};
    var td = new TextDecoder("utf-8");
    while (i < b.length) {
      var nl = dv.getUint16(i, true); i += 2;
      var name = td.decode(b.subarray(i, i + nl)); i += nl;
      var kind = b[i]; i += 1;
      var len = dv.getUint32(i, true) + dv.getUint32(i + 4, true) * 4294967296; i += 8;
      if (kind === 1) out[name] = uvarints(b, i, len);
      else if (kind === 2) out[name] = zigzag(uvarints(b, i, len));
      else if (kind === 3) out[name] = JSON.parse(td.decode(b.subarray(i, i + len)));
      else if (kind === 4) out[name] = len ? td.decode(b.subarray(i, i + len)).split("\0") : [];
      else out[name] = b.slice(i, i + len);
      i += len;
    }
    return out;
  }
  function cumsum(a, div) {
    var o = new Float64Array(a.length), s = 0;
    for (var i = 0; i < a.length; i++) { s += a[i]; o[i] = div ? s / div : s; }
    return o;
  }
  function offsets(cnt, mul) {
    var o = new Int32Array(cnt.length + 1);
    for (var i = 0; i < cnt.length; i++) o[i + 1] = o[i] + cnt[i] * (mul || 1);
    return o;
  }

  // ---------------- χάρτης ----------------
  function AtticaMap(d) {
    var sc = d.meta.scale, i, e;
    this.meta = d.meta; this.S = d.str; this.F = d.meta.flags; this.HW = d.meta.highway_classes;
    var vx = cumsum(d["v.dx"]), vy = cumsum(d["v.dy"]);
    var nv = vx.length;
    this.nv = nv;
    this.vlon = new Float64Array(nv); this.vlat = new Float64Array(nv);
    for (i = 0; i < nv; i++) { this.vlon[i] = vx[i] / sc; this.vlat[i] = vy[i] / sc; }
    var du = d["e.u"], dvv = d["e.v"], ne = du.length;
    this.ne = ne;
    this.eu = new Int32Array(ne); this.ev = new Int32Array(ne);
    var pv = 0;
    for (e = 0; e < ne; e++) { var u = pv + du[e]; this.eu[e] = u; this.ev[e] = u + dvv[e]; pv = this.ev[e]; }
    this.elen = new Float32Array(ne); this.ecls = new Uint8Array(ne); this.eflags = new Uint16Array(ne);
    this.espeed = new Uint8Array(ne); this.eway = new Int32Array(ne);
    var wsum = 0;
    for (e = 0; e < ne; e++) {
      this.elen[e] = d["e.len"][e] / 10; this.ecls[e] = d["e.cls"][e]; this.eflags[e] = d["e.flags"][e];
      this.espeed[e] = d["e.speed"][e]; wsum += d["e.way"][e]; this.eway[e] = wsum;
    }
    this.eoff = offsets(d["e.ni"]);
    var ng = this.eoff[ne];
    this.glon = new Float64Array(ng); this.glat = new Float64Array(ng);
    var gdx = d["g.dx"], gdy = d["g.dy"];
    for (e = 0; e < ne; e++) {
      var x = vx[this.eu[e]], y = vy[this.eu[e]];
      for (var k = this.eoff[e]; k < this.eoff[e + 1]; k++) {
        x += gdx[k]; y += gdy[k]; this.glon[k] = x / sc; this.glat[k] = y / sc;
      }
    }
    this.wayOff = offsets(d["way.nt"], 2); this.wayKv = new Int32Array(d["way.kv"]);
    var px = cumsum(d["p.dx"]), py = cumsum(d["p.dy"]), np_ = px.length;
    this.np = np_;
    this.plon = new Float64Array(np_); this.plat = new Float64Array(np_);
    for (i = 0; i < np_; i++) { this.plon[i] = px[i] / sc; this.plat[i] = py[i] / sc; }
    this.pOff = offsets(d["p.nt"], 2); this.pKv = new Int32Array(d["p.kv"]);
    // όριο: μία συνεχής σειρά διαφορών
    var bn = d["bnd.n"], bx = cumsum(d["bnd.dx"]), by = cumsum(d["bnd.dy"]), o = 0;
    this.boundary = [];
    for (i = 0; i < bn.length; i++) {
      var ring = new Float64Array(bn[i] * 2);
      for (k = 0; k < bn[i]; k++) { ring[2 * k] = bx[o + k] / sc; ring[2 * k + 1] = by[o + k] / sc; }
      this.boundary.push(ring); o += bn[i];
    }
    this.routes = d.transit.routes; this.masters = d.transit.masters;
    var rn = d["rs.n"], rdx = d["rs.dx"], rdy = d["rs.dy"]; o = 0;
    for (i = 0; i < this.routes.length; i++) {
      var n = rn[i], sx = new Float64Array(n), sy = new Float64Array(n), ax = 0, ay = 0;
      for (k = 0; k < n; k++) { ax += rdx[o + k]; ay += rdy[o + k]; sx[k] = ax / sc; sy[k] = ay / sc; }
      this.routes[i].shx = sx; this.routes[i].shy = sy; o += n;
    }
    this.wv = d.wv;
    this.graphs = {};
    this.vgrid = new Grid(this.vlon, this.vlat, 0.004);
    this.ggrid = new Grid(this.glon, this.glat, 0.004);
    this.gedge = new Int32Array(ng);
    for (e = 0; e < ne; e++) for (k = this.eoff[e]; k < this.eoff[e + 1]; k++) this.gedge[k] = e;
    this._stopFlag = null;
    this.index = null;
  }

  AtticaMap.prototype.wayTags = function (w) {
    var t = {}, a = this.wayOff[w], b = this.wayOff[w + 1];
    for (var i = a; i < b; i += 2) t[this.S[this.wayKv[i]]] = this.S[this.wayKv[i + 1]];
    return t;
  };
  AtticaMap.prototype.poiTags = function (p) {
    var t = {}, a = this.pOff[p], b = this.pOff[p + 1];
    for (var i = a; i < b; i += 2) t[this.S[this.pKv[i]]] = this.S[this.pKv[i + 1]];
    return t;
  };
  AtticaMap.prototype.isStop = function (p) {
    if (!this._stopFlag) {
      var f = new Uint8Array(this.np), S = this.S;
      for (var q = 0; q < this.np; q++) {
        for (var i = this.pOff[q]; i < this.pOff[q + 1]; i += 2) {
          var k = S[this.pKv[i]], v = S[this.pKv[i + 1]];
          if (k === "public_transport" || k === "railway" || (k === "highway" && v === "bus_stop")) { f[q] = 1; break; }
        }
      }
      this._stopFlag = f;
    }
    return this._stopFlag[p] === 1;
  };
  AtticaMap.prototype.poiName = function (p, t) {
    t = t || this.poiTags(p);
    var n = t.name || t["name:el"] || t["name:en"] ||
      ((t["addr:street"] || "") + " " + (t["addr:housenumber"] || "")).trim();
    if (n) return n;
    var best = null, bd = 150;
    for (var q = 0; q < this.np; q++) {
      if (Math.abs(this.plat[q] - this.plat[p]) > 0.002 || Math.abs(this.plon[q] - this.plon[p]) > 0.002 || q === p) continue;
      var d = hav(this.plat[p], this.plon[p], this.plat[q], this.plon[q]);
      if (d < bd && this.isStop(q)) { var tq = this.poiTags(q); if (tq.name) { best = tq.name; bd = d; } }
    }
    return best || t.ref || ("POI " + p);
  };
  AtticaMap.prototype.edgeCoords = function (e, rev) {
    var lon = [this.vlon[this.eu[e]]], lat = [this.vlat[this.eu[e]]];
    for (var k = this.eoff[e]; k < this.eoff[e + 1]; k++) { lon.push(this.glon[k]); lat.push(this.glat[k]); }
    lon.push(this.vlon[this.ev[e]]); lat.push(this.vlat[this.ev[e]]);
    if (rev) { lon.reverse(); lat.reverse(); }
    return [lon, lat];
  };

  // ---------------- χωρικό ευρετήριο (πλέγμα) ----------------
  function Grid(lon, lat, cell) {
    this.cell = cell; this.lon = lon; this.lat = lat;
    var n = lon.length, keys = new Float64Array(n), i;
    var x0 = Infinity, y0 = Infinity;
    for (i = 0; i < n; i++) { if (lon[i] < x0) x0 = lon[i]; if (lat[i] < y0) y0 = lat[i]; }
    this.x0 = x0; this.y0 = y0; this.W = 100000;
    var idx = new Int32Array(n);
    for (i = 0; i < n; i++) { keys[i] = this.key(lon[i], lat[i]); idx[i] = i; }
    var arr = Array.prototype.slice.call(idx);
    arr.sort(function (a, b) { return keys[a] - keys[b]; });
    this.ids = new Int32Array(arr);
    this.cells = new Map();
    var start = 0;
    for (i = 1; i <= n; i++) {
      if (i === n || keys[this.ids[i]] !== keys[this.ids[start]]) {
        this.cells.set(keys[this.ids[start]], [start, i]); start = i;
      }
    }
  }
  Grid.prototype.key = function (lo, la) {
    return Math.floor((lo - this.x0) / this.cell) * this.W + Math.floor((la - this.y0) / this.cell);
  };
  // κοντινότερο στοιχείο που περνά το φίλτρο, μέχρι maxRing κελιά μακριά
  Grid.prototype.nearest = function (lo, la, ok, maxRing) {
    var cx = Math.floor((lo - this.x0) / this.cell), cy = Math.floor((la - this.y0) / this.cell);
    var best = -1, bd = Infinity, c = Math.cos(la * D2R);
    for (var r = 0; r <= (maxRing || 60); r++) {
      for (var dx = -r; dx <= r; dx++) {
        for (var dy = -r; dy <= r; dy++) {
          if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
          var cl = this.cells.get((cx + dx) * this.W + (cy + dy));
          if (!cl) continue;
          for (var j = cl[0]; j < cl[1]; j++) {
            var i = this.ids[j];
            if (ok && !ok(i)) continue;
            var a = (this.lon[i] - lo) * c, b = this.lat[i] - la, dd = a * a + b * b;
            if (dd < bd) { bd = dd; best = i; }
          }
        }
      }
      // αν βρέθηκε κάτι και ο επόμενος δακτύλιος είναι σίγουρα πιο μακριά, σταμάτα
      if (best >= 0 && Math.sqrt(bd) < r * this.cell * Math.min(c, 1)) break;
    }
    return best;
  };

  // ---------------- γράφοι ----------------
  AtticaMap.prototype.graph = function (mode, metric) {
    metric = metric || "time";
    var key = mode + "|" + metric;
    if (this.graphs[key]) return this.graphs[key];
    var F = this.F, ne = this.ne, nv = this.nv, e;
    var fw = new Uint8Array(ne), bw = new Uint8Array(ne), ew = new Float64Array(ne);
    var steps = this.HW.indexOf("steps");
    for (e = 0; e < ne; e++) {
      var fl = this.eflags[e], L = this.elen[e], ferry = fl & F.ferry, w;
      var fsp = Math.max(this.espeed[e], 5) / 3.6;
      if (mode === "car") { fw[e] = (fl & F.car_f) ? 1 : 0; bw[e] = (fl & F.car_b) ? 1 : 0; w = L / fsp; }
      else if (mode === "bike") { fw[e] = (fl & F.bike_f) ? 1 : 0; bw[e] = (fl & F.bike_b) ? 1 : 0; w = L / (ferry ? fsp : BIKE); }
      else { fw[e] = bw[e] = (fl & F.foot) ? 1 : 0; w = L / (ferry ? fsp : (this.ecls[e] === steps ? WALK / 1.6 : WALK)); }
      if (ferry) w += FERRY_WAIT;
      ew[e] = metric === "dist" ? L : w;
    }
    // συνεκτικές συνιστώσες (union-find)
    var par = new Int32Array(nv);
    for (var i = 0; i < nv; i++) par[i] = i;
    function find(x) { while (par[x] !== x) { par[x] = par[par[x]]; x = par[x]; } return x; }
    for (e = 0; e < ne; e++) if (fw[e] || bw[e]) {
      var ra = find(this.eu[e]), rb = find(this.ev[e]); if (ra !== rb) par[ra] = rb;
    }
    var cnt = new Int32Array(nv);
    for (i = 0; i < nv; i++) cnt[find(i)]++;
    var big = new Uint8Array(nv);
    for (i = 0; i < nv; i++) big[i] = cnt[find(i)] >= 200 ? 1 : 0;
    var g = { mode: mode, metric: metric, nv: nv, fw: fw, bw: bw, ew: ew, big: big, extra: null };
    // ακμές ως λίστες (src, dst, w, eid)
    var src = [], dst = [], wt = [], eid = [];
    for (e = 0; e < ne; e++) {
      if (fw[e]) { src.push(this.eu[e]); dst.push(this.ev[e]); wt.push(ew[e]); eid.push(e); }
      if (bw[e]) { src.push(this.ev[e]); dst.push(this.eu[e]); wt.push(ew[e]); eid.push(-e - 1); }
    }
    if (mode === "transit") this.addTransit(g, src, dst, wt, eid);
    var n = nv + (g.extra ? g.extra.info.length : 0), m = src.length;
    var ptr = new Int32Array(n + 1);
    for (i = 0; i < m; i++) ptr[src[i] + 1]++;
    for (i = 0; i < n; i++) ptr[i + 1] += ptr[i];
    var pos = ptr.slice(0, n), D = new Int32Array(m), Wt = new Float64Array(m), E = new Int32Array(m), Sr = new Int32Array(m);
    for (i = 0; i < m; i++) {
      var p = pos[src[i]]++; D[p] = dst[i]; Wt[p] = wt[i]; E[p] = eid[i]; Sr[p] = src[i];
    }
    g.n = n; g.ptr = ptr; g.dst = D; g.w = Wt; g.eid = E; g.src = Sr;
    this.graphs[key] = g;
    return g;
  };

  AtticaMap.prototype.addTransit = function (g, src, dst, wt, eid) {
    var self = this, nv = this.nv, info = [], cache = {};
    var okv = new Uint8Array(nv);
    for (var e = 0; e < this.ne; e++) if (g.fw[e] && g.big[this.eu[e]]) { okv[this.eu[e]] = 1; okv[this.ev[e]] = 1; }
    function access(p) {
      if (cache[p] === undefined) {
        var v = self.vgrid.nearest(self.plon[p], self.plat[p], function (i) { return okv[i] === 1; }, 4);
        var d = v >= 0 ? hav(self.plat[p], self.plon[p], self.vlat[v], self.vlon[v]) : Infinity;
        cache[p] = d <= ACCESS_MAX ? [v, d] : null;
      }
      return cache[p];
    }
    for (var ri = 0; ri < this.routes.length; ri++) {
      var r = this.routes[ri], t = r.tags, kind = t.route || "bus";
      var par = TRANSIT_PARAMS[kind] || [18, 600], spd = par[0], wait = par[1];
      if (t.interval) {
        var mm = /^(?:(\d+):)?(\d+)/.exec(t.interval);
        if (mm) { var mins = (+(mm[1] || 0)) * 60 + (+mm[2]); if (mins > 0 && mins < 300) wait = mins * 30; }
      }
      var seq = [];
      for (var j = 0; j < r.stops.length; j++) {
        var p = r.stops[j][0], role = r.stops[j][1];
        if (seq.length) {
          var q = seq[seq.length - 1];
          if (hav(this.plat[p], this.plon[p], this.plat[q], this.plon[q]) < 80) {
            if (role.indexOf("platform") === 0 && !this.isPlatform(q)) seq[seq.length - 1] = p;
            continue;
          }
        }
        seq.push(p);
      }
      seq = seq.filter(function (p) { return access(p) !== null; });
      if (seq.length < 2) continue;
      var shx = r.shx, shy = r.shy, cum = null, kk = null;
      if (shx.length > 1) {
        cum = new Float64Array(shx.length);
        for (var s = 1; s < shx.length; s++) cum[s] = cum[s - 1] + hav(shy[s - 1], shx[s - 1], shy[s], shx[s]);
        kk = seq.map(function (p) {
          var b = 0, bd = Infinity, c = Math.cos(self.plat[p] * D2R);
          for (var s = 0; s < shx.length; s++) {
            var a1 = (shx[s] - self.plon[p]) * c, a2 = shy[s] - self.plat[p], dd = a1 * a1 + a2 * a2;
            if (dd < bd) { bd = dd; b = s; }
          }
          return b;
        });
      }
      var both = t["public_transport:version"] === "1", base = nv + info.length;
      for (j = 0; j < seq.length; j++) {
        info.push([ri, j, seq[j]]);
        var ac = access(seq[j]), rv = base + j;
        src.push(ac[0], rv); dst.push(rv, ac[0]); wt.push(wait + ac[1] / WALK, ac[1] / WALK); eid.push(TRANSIT_EDGE, TRANSIT_EDGE);
      }
      for (j = 0; j + 1 < seq.length; j++) {
        var a = seq[j], b2 = seq[j + 1];
        var hd = hav(this.plat[a], this.plon[a], this.plat[b2], this.plon[b2]), dd2 = hd * 1.25;
        if (cum && kk[j + 1] > kk[j]) {
          var sd = cum[kk[j + 1]] - cum[kk[j]];
          if (sd >= hd * 0.9 && sd <= hd * 4) dd2 = sd;
        }
        var tt = dd2 / (spd / 3.6) + ((kind === "bus" || kind === "trolleybus" || kind === "tram") ? 20 : 30);
        src.push(base + j); dst.push(base + j + 1); wt.push(tt); eid.push(TRANSIT_EDGE);
        if (both) { src.push(base + j + 1); dst.push(base + j); wt.push(tt); eid.push(TRANSIT_EDGE); }
      }
    }
    g.extra = { info: info };
  };
  AtticaMap.prototype.isPlatform = function (p) {
    var t = this.poiTags(p);
    return t.public_transport === "platform" || t.highway === "bus_stop";
  };

  // ---------------- snap ----------------
  AtticaMap.prototype.snap = function (lat, lon, g) {
    var self = this;
    var okE = function (e) { return (g.fw[e] || g.bw[e]) && g.big[self.eu[e]]; };
    var hasBig = false;
    for (var e = 0; e < this.ne && !hasBig; e++) if (okE(e)) hasBig = true;
    if (!hasBig) okE = function (e) { return g.fw[e] || g.bw[e]; };
    var okv = g._okv;
    if (!okv) {
      okv = new Uint8Array(this.nv);
      for (e = 0; e < this.ne; e++) if (okE(e)) { okv[this.eu[e]] = 1; okv[this.ev[e]] = 1; }
      g._okv = okv;
    }
    var v = this.vgrid.nearest(lon, lat, function (i) { return okv[i] === 1; });
    var gp = this.ggrid.nearest(lon, lat, function (k) { return okE(self.gedge[k]); });
    var c = Math.cos(lat * D2R);
    function d2(x, y) { var a = (x - lon) * c, b = y - lat; return a * a + b * b; }
    if (gp >= 0 && (v < 0 || d2(this.glon[gp], this.glat[gp]) < d2(this.vlon[v], this.vlat[v]))) {
      var ed = this.gedge[gp], cc = this.edgeCoords(ed), lo = cc[0], la = cc[1], cum = 0, s = 0;
      var j = gp - this.eoff[ed] + 1;
      for (var k = 1; k < lo.length; k++) {
        cum += hav(la[k - 1], lo[k - 1], la[k], lo[k]);
        if (k === j) s = cum;
      }
      return { kind: "e", e: ed, fr: cum > 0 ? s / cum : 0, d: hav(lat, lon, this.glat[gp], this.glon[gp]) };
    }
    return { kind: "v", v: v, d: v >= 0 ? hav(lat, lon, this.vlat[v], this.vlon[v]) : Infinity };
  };
  AtticaMap.prototype.ends = function (g, sn, asTarget) {
    var out = {};
    if (sn.kind === "v") { out[sn.v] = 0; return out; }
    var e = sn.e, fr = sn.fr, w = g.ew[e], u = this.eu[e], v = this.ev[e];
    if (asTarget) {
      if (g.fw[e]) out[u] = w * fr;
      if (g.bw[e]) out[v] = Math.min(out[v] === undefined ? Infinity : out[v], w * (1 - fr));
    } else {
      if (g.fw[e]) out[v] = w * (1 - fr);
      if (g.bw[e]) out[u] = Math.min(out[u] === undefined ? Infinity : out[u], w * fr);
    }
    return out;
  };

  // ---------------- Dijkstra (binary heap) ----------------
  function Heap(cap) { this.k = new Float64Array(cap); this.v = new Int32Array(cap); this.n = 0; }
  Heap.prototype.push = function (key, val) {
    if (this.n === this.k.length) {
      var k2 = new Float64Array(this.n * 2), v2 = new Int32Array(this.n * 2);
      k2.set(this.k); v2.set(this.v); this.k = k2; this.v = v2;
    }
    var i = this.n++, K = this.k, V = this.v;
    while (i > 0) {
      var p = (i - 1) >> 1;
      if (K[p] <= key) break;
      K[i] = K[p]; V[i] = V[p]; i = p;
    }
    K[i] = key; V[i] = val;
  };
  Heap.prototype.pop = function () {
    var K = this.k, V = this.v, topK = K[0], topV = V[0];
    var n = --this.n, lk = K[n], lv = V[n], i = 0;
    while (true) {
      var l = 2 * i + 1;
      if (l >= n) break;
      var r = l + 1, c = (r < n && K[r] < K[l]) ? r : l;
      if (K[c] >= lk) break;
      K[i] = K[c]; V[i] = V[c]; i = c;
    }
    K[i] = lk; V[i] = lv;
    this.lastKey = topK;
    return topV;
  };

  AtticaMap.prototype.dijkstra = function (g, sources, targets) {
    var n = g.n, dist = new Float64Array(n), pred = new Int32Array(n), done = new Uint8Array(n);
    dist.fill(Infinity); pred.fill(-2);
    var h = new Heap(1024), v, best = Infinity, bestT = -1, settled = 0;
    for (var s in sources) { v = +s; if (sources[s] < dist[v]) { dist[v] = sources[s]; pred[v] = -1; h.push(sources[s], v); } }
    var ptr = g.ptr, D = g.dst, W = g.w;
    while (h.n) {
      v = h.pop(); var c = h.lastKey;
      if (done[v]) continue;
      if (c >= best) break;
      done[v] = 1; settled++;
      var tv = targets[v];
      if (tv !== undefined && c + tv < best) { best = c + tv; bestT = v; }
      for (var i = ptr[v], e = ptr[v + 1]; i < e; i++) {
        var x = D[i], nc = c + W[i];
        if (nc < dist[x]) { dist[x] = nc; pred[x] = i; h.push(nc, x); }
      }
    }
    if (bestT < 0) return null;
    var path = [];
    v = bestT;
    while (pred[v] >= 0) { path.push(pred[v]); v = g.src[pred[v]]; }
    path.reverse();
    return { cost: best, path: path, settled: settled };
  };

  AtticaMap.prototype.route = function (a, b, mode, metric) {
    var t0 = performance.now();
    var g = this.graph(mode, metric);
    var tg = performance.now() - t0;
    var sa = this.snap(a.lat, a.lon, g), sb = this.snap(b.lat, b.lon, g);
    var t1 = performance.now();
    var res = this.dijkstra(g, this.ends(g, sa, false), this.ends(g, sb, true));
    var dt = performance.now() - t1;
    if (!res) return null;
    var out = this.explain(g, res.path);
    out.cost = res.cost; out.settled = res.settled; out.ms = dt; out.graphMs = tg;
    out.snap = [sa.d, sb.d]; out.mode = mode; out.metric = metric || "time";
    return out;
  };

  AtticaMap.prototype.routeLabel = function (r) {
    var t = r.tags, kind = TYPE_EL[t.route] || t.route || "";
    return (kind + " " + (t.ref || "") + (t.name ? " (" + t.name + ")" : "")).replace(/\s+/g, " ").trim();
  };

  AtticaMap.prototype.explain = function (g, path) {
    var steps = [], segs = [], cur = null, dist = 0, nv = g.nv, self = this;
    function flush() { if (cur) steps.push(cur); }
    for (var k = 0; k < path.length; k++) {
      var i = path[k], s = g.src[i], d = g.dst[i], e = g.eid[i];
      if (e !== TRANSIT_EDGE) {
        var ed = e >= 0 ? e : -e - 1, cc = this.edgeCoords(ed, e < 0);
        dist += this.elen[ed];
        var wt = this.wayTags(this.eway[ed]);
        var ferry = this.eflags[ed] & this.F.ferry;
        var name = wt.name || wt.ref || (ferry ? "πλοίο" : this.HW[this.ecls[ed]]);
        var kind = (g.mode === "foot" || g.mode === "transit") ? "walk" : g.mode;
        segs.push({ lon: cc[0], lat: cc[1], kind: ferry ? "ferry" : kind });
        if (cur && cur.type === kind && cur.name === name) { cur.dist += this.elen[ed]; cur.time += g.w[i]; }
        else { flush(); cur = { type: kind, name: name, dist: this.elen[ed], time: g.w[i] }; }
      } else {
        var info = g.extra.info;
        if (s < nv && d >= nv) {
          var it = info[d - nv]; flush();
          cur = { type: "ride", route: it[0], from: it[2], to: it[2], stops: 0, time: g.w[i], dist: 0,
            name: this.routeLabel(this.routes[it[0]]), kindr: this.routes[it[0]].tags.route,
            lon: [this.plon[it[2]]], lat: [this.plat[it[2]]] };
        } else if (s >= nv && d >= nv) {
          var p = info[d - nv][2], q = info[s - nv][2];
          cur.stops++; cur.to = p; cur.time += g.w[i];
          cur.dist += hav(this.plat[q], this.plon[q], this.plat[p], this.plon[p]);
          cur.lon.push(this.plon[p]); cur.lat.push(this.plat[p]);
        } else {
          cur.time += g.w[i];
          segs.push({ lon: cur.lon, lat: cur.lat, kind: "ride", rk: cur.kindr });
          flush(); cur = null;
        }
      }
    }
    flush();
    return { steps: steps, segs: segs, dist: dist };
  };

  // ---------------- αναζήτηση ----------------
  AtticaMap.prototype.buildIndex = function () {
    var keys = ["name", "name:el", "name:en", "alt_name", "official_name", "short_name", "old_name", "int_name", "loc_name", "ref"];
    var want = {}, i, S = this.S;
    for (i = 0; i < S.length; i++) if (keys.indexOf(S[i]) >= 0) want[i] = S[i];
    var names = [], kind = [], ref = [];
    for (var p = 0; p < this.np; p++) {
      for (i = this.pOff[p]; i < this.pOff[p + 1]; i += 2) {
        var k = want[this.pKv[i]];
        if (k === undefined) continue;
        if (k === "ref" && !this.isStop(p)) continue;
        names.push(norm(S[this.pKv[i + 1]])); kind.push("p"); ref.push(p);
      }
    }
    var A = this.wv.articles, L = this.wv.listings;
    for (i = 0; i < A.length; i++) { names.push(norm(A[i].title)); kind.push("a"); ref.push(i); }
    for (i = 0; i < L.length; i++) if (L[i].name) { names.push(norm(L[i].name)); kind.push("l"); ref.push(i); }
    this.index = { names: names, kind: kind, ref: ref };
  };
  AtticaMap.prototype.rank = function (k, r) {
    if (k === "a") return 0;
    if (k === "l") return 6;
    var t = this.poiTags(r), pl = t.place;
    if (pl === "city" || pl === "town" || pl === "municipality") return 1;
    if (t.railway === "station" || t.public_transport === "station" || t.aeroway === "aerodrome" || t.amenity === "ferry_terminal") return 2;
    if (["suburb", "village", "quarter", "neighbourhood", "island"].indexOf(pl) >= 0) return 3;
    if (t.tourism || t.historic || t.amenity === "university" || t.amenity === "hospital") return 4;
    return 5;
  };
  AtticaMap.prototype.describe = function (k, r) {
    if (k === "p") {
      var t = this.poiTags(r), typ = "";
      var tk = ["place", "railway", "public_transport", "amenity", "shop", "tourism", "historic", "leisure", "highway", "natural", "office", "aeroway"];
      for (var i = 0; i < tk.length; i++) if (t[tk[i]]) { typ = tk[i] + "=" + t[tk[i]]; break; }
      return { kind: "poi", id: r, name: this.poiName(r, t), type: typ, lat: this.plat[r], lon: this.plon[r] };
    }
    if (k === "a") { var a = this.wv.articles[r]; return { kind: "article", id: r, name: a.title, type: "Wikivoyage", lat: a.lat, lon: a.lon }; }
    var l = this.wv.listings[r];
    return { kind: "listing", id: r, name: l.name, type: "Wikivoyage: " + (l.sec || ""), lat: l.lat, lon: l.lon };
  };
  AtticaMap.prototype.search = function (text, limit) {
    var m = /^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$/.exec(text);
    if (m) return [{ kind: "coord", name: text.trim(), type: "συντεταγμένες", lat: +m[1], lon: +m[2] }];
    var q = norm(text);
    if (!q) return [];
    if (!this.index) this.buildIndex();
    var N = this.index.names, hits = [];
    for (var i = 0; i < N.length; i++) {
      var n = N[i], tier;
      if (n === q) tier = 0;
      else if (n.indexOf(q + " ") === 0 || n.slice(-q.length - 1) === " " + q) tier = 1;
      else if (n.indexOf(q) >= 0) tier = 2;
      else continue;
      hits.push([tier, 0, n.length, i]);
    }
    hits.sort(function (a, b) { return a[0] - b[0] || a[2] - b[2]; });
    hits = hits.slice(0, 400);
    var self = this;
    hits.forEach(function (h) { h[1] = self.rank(self.index.kind[h[3]], self.index.ref[h[3]]); });
    hits.sort(function (a, b) { return a[0] - b[0] || a[1] - b[1] || a[2] - b[2]; });
    var out = [], seen = {};
    for (i = 0; i < hits.length && out.length < (limit || 10); i++) {
      var k = this.index.kind[hits[i][3]], r = this.index.ref[hits[i][3]], key = k + r;
      if (seen[key]) continue;
      seen[key] = 1;
      out.push(this.describe(k, r));
    }
    return out;
  };

  // ================= UI =================
  var M = null, cv = document.getElementById("map"), ctx = cv.getContext("2d");
  var view = { cx: 23.73, cy: 37.98, s: 3000 };   // s: pixels ανά μοίρα (lat)
  var DPR = Math.min(window.devicePixelRatio || 1, 2);
  var sel = { from: null, to: null }, result = null, pickStage = 0, dirty = true;
  var STY = {}, ORDER = null, EB = null;

  function setLoad(t, f) {
    document.getElementById("ltxt").textContent = t;
    if (f !== undefined) document.querySelector("#bar i").style.width = Math.round(f * 100) + "%";
  }
  function resize() {
    cv.width = Math.floor(innerWidth * DPR); cv.height = Math.floor(innerHeight * DPR);
    cv.style.width = innerWidth + "px"; cv.style.height = innerHeight + "px"; dirty = true;
  }
  function proj(lon, lat) {
    var c = Math.cos(view.cy * D2R);
    return [cv.width / 2 + (lon - view.cx) * c * view.s * DPR, cv.height / 2 - (lat - view.cy) * view.s * DPR];
  }
  function unproj(x, y) {
    var c = Math.cos(view.cy * D2R);
    return [view.cx + (x * DPR - cv.width / 2) / (c * view.s * DPR), view.cy - (y * DPR - cv.height / 2) / (view.s * DPR)];
  }

  function prepStyles() {
    var S = {
      motorway: ["#ff8c3c", 3, 9], motorway_link: ["#ff8c3c", 2, 8], trunk: ["#fabe46", 3, 8], trunk_link: ["#fabe46", 2, 7],
      primary: ["#ebd26e", 2.5, 7], primary_link: ["#ebd26e", 1.5, 6], secondary: ["#c8c8aa", 2, 6], secondary_link: ["#c8c8aa", 1.2, 5],
      tertiary: ["#969ba0", 1.5, 5], tertiary_link: ["#969ba0", 1, 4], ferry: ["#5096ff", 1.5, 9]
    };
    var foot = ["footway", "path", "cycleway", "bridleway", "steps", "pedestrian", "corridor"];
    M.HW.forEach(function (h, i) { STY[i] = S[h] || (foot.indexOf(h) >= 0 ? ["#30343c", 0.8, 0] : ["#464c56", 1, 2]); });
    var ne = M.ne, idx = [];
    for (var e = 0; e < ne; e++) idx.push(e);
    idx.sort(function (a, b) { return STY[M.ecls[a]][2] - STY[M.ecls[b]][2]; });
    ORDER = new Int32Array(idx);
    EB = new Float32Array(ne * 4);
    for (e = 0; e < ne; e++) {
      var x0 = Math.min(M.vlon[M.eu[e]], M.vlon[M.ev[e]]), x1 = Math.max(M.vlon[M.eu[e]], M.vlon[M.ev[e]]);
      var y0 = Math.min(M.vlat[M.eu[e]], M.vlat[M.ev[e]]), y1 = Math.max(M.vlat[M.eu[e]], M.vlat[M.ev[e]]);
      for (var k = M.eoff[e]; k < M.eoff[e + 1]; k++) {
        if (M.glon[k] < x0) x0 = M.glon[k]; if (M.glon[k] > x1) x1 = M.glon[k];
        if (M.glat[k] < y0) y0 = M.glat[k]; if (M.glat[k] > y1) y1 = M.glat[k];
      }
      EB[4 * e] = x0; EB[4 * e + 1] = y0; EB[4 * e + 2] = x1; EB[4 * e + 3] = y1;
    }
  }

  function draw() {
    dirty = false;
    ctx.fillStyle = "#0e1117"; ctx.fillRect(0, 0, cv.width, cv.height);
    if (!M) return;
    var a = unproj(0, innerHeight), b = unproj(innerWidth, 0);
    var minRank = view.s < 600 ? 7 : view.s < 1500 ? 5 : view.s < 6000 ? 2 : -1;
    var lw = Math.max(0.6, Math.min(2.5, view.s / 8000)) * DPR;
    // όριο
    ctx.strokeStyle = "#5a6478"; ctx.lineWidth = DPR;
    M.boundary.forEach(function (r) {
      ctx.beginPath();
      for (var k = 0; k < r.length; k += 2) { var p = proj(r[k], r[k + 1]); if (k) ctx.lineTo(p[0], p[1]); else ctx.moveTo(p[0], p[1]); }
      ctx.stroke();
    });
    var curSty = -1;
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    for (var j = 0; j < ORDER.length; j++) {
      var e = ORDER[j], st = STY[M.ecls[e]];
      if (st[2] < minRank) continue;
      if (EB[4 * e + 2] < a[0] || EB[4 * e] > b[0] || EB[4 * e + 3] < a[1] || EB[4 * e + 1] > b[1]) continue;
      if (M.ecls[e] !== curSty) {
        if (curSty >= 0) ctx.stroke();
        curSty = M.ecls[e]; ctx.strokeStyle = st[0]; ctx.lineWidth = st[1] * lw; ctx.beginPath();
      }
      var p = proj(M.vlon[M.eu[e]], M.vlat[M.eu[e]]); ctx.moveTo(p[0], p[1]);
      for (var k = M.eoff[e]; k < M.eoff[e + 1]; k++) { p = proj(M.glon[k], M.glat[k]); ctx.lineTo(p[0], p[1]); }
      p = proj(M.vlon[M.ev[e]], M.vlat[M.ev[e]]); ctx.lineTo(p[0], p[1]);
    }
    if (curSty >= 0) ctx.stroke();
    // στάσεις σε μεγάλο zoom
    if (view.s > 20000) {
      ctx.fillStyle = "#ffffff";
      for (var q = 0; q < M.np; q++) {
        if (M.plon[q] < a[0] || M.plon[q] > b[0] || M.plat[q] < a[1] || M.plat[q] > b[1] || !M.isStop(q)) continue;
        var pp = proj(M.plon[q], M.plat[q]); ctx.fillRect(pp[0] - 2 * DPR, pp[1] - 2 * DPR, 4 * DPR, 4 * DPR);
      }
    }
    // διαδρομή
    if (result) {
      for (var pass = 0; pass < 2; pass++) {   // πρώτα όλα τα περιγράμματα, μετά τα χρώματα
        result.segs.forEach(function (s) {
          var col = s.kind === "ride" ? (s.rk === "subway" ? "#3cc878" : s.rk === "train" ? "#e650c8" : s.rk === "tram" ? "#5adce6" : s.rk === "ferry" ? "#5096ff" : "#ff9a3c")
            : s.kind === "ferry" ? "#5096ff" : s.kind === "walk" ? "#bbbbbb" : "#4fc3f7";
          ctx.strokeStyle = pass ? col : "#000"; ctx.lineWidth = (pass ? 5 : 8) * DPR;
          if (s.kind === "walk" && pass) ctx.setLineDash([6 * DPR, 6 * DPR]); else ctx.setLineDash([]);
          ctx.beginPath();
          for (var k = 0; k < s.lon.length; k++) { var p = proj(s.lon[k], s.lat[k]); if (k) ctx.lineTo(p[0], p[1]); else ctx.moveTo(p[0], p[1]); }
          ctx.stroke();
        });
      }
      ctx.setLineDash([]);
    }
    [["from", "#3cdc78"], ["to", "#f05050"]].forEach(function (x) {
      var s = sel[x[0]]; if (!s) return;
      var p = proj(s.lon, s.lat);
      ctx.fillStyle = x[1]; ctx.strokeStyle = "#fff"; ctx.lineWidth = 2 * DPR;
      ctx.beginPath(); ctx.arc(p[0], p[1], 8 * DPR, 0, 7); ctx.fill(); ctx.stroke();
    });
  }
  function loop() { if (dirty) draw(); requestAnimationFrame(loop); }

  // ---- pan / zoom ----
  function zoomAt(f, x, y) {
    var before = unproj(x, y);
    view.s = Math.max(150, Math.min(400000, view.s * f));
    var after = unproj(x, y);
    view.cx += before[0] - after[0]; view.cy += before[1] - after[1]; dirty = true;
  }
  var ptrs = {}, lastDist = 0, moved = false, downAt = null;
  cv.addEventListener("pointerdown", function (ev) {
    cv.setPointerCapture(ev.pointerId); ptrs[ev.pointerId] = [ev.clientX, ev.clientY]; moved = false; downAt = [ev.clientX, ev.clientY];
  });
  cv.addEventListener("pointermove", function (ev) {
    if (!ptrs[ev.pointerId]) return;
    var ids = Object.keys(ptrs), old = ptrs[ev.pointerId];
    if (ids.length === 1) {
      var c = Math.cos(view.cy * D2R);
      view.cx -= (ev.clientX - old[0]) / (c * view.s); view.cy += (ev.clientY - old[1]) / view.s; dirty = true;
      if (Math.abs(ev.clientX - downAt[0]) + Math.abs(ev.clientY - downAt[1]) > 6) moved = true;
    }
    ptrs[ev.pointerId] = [ev.clientX, ev.clientY];
    if (ids.length === 2) {
      var p1 = ptrs[ids[0]], p2 = ptrs[ids[1]], dd = Math.hypot(p1[0] - p2[0], p1[1] - p2[1]);
      if (lastDist) zoomAt(dd / lastDist, (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2);
      lastDist = dd; moved = true;
    }
  });
  function up(ev) {
    delete ptrs[ev.pointerId]; lastDist = 0;
    if (!moved && pickStage && M && ev.type === "pointerup") {
      var ll = unproj(ev.clientX, ev.clientY), pt = { kind: "coord", name: ll[1].toFixed(5) + "," + ll[0].toFixed(5), lat: ll[1], lon: ll[0] };
      if (pickStage === 1) { setSel("from", pt); pickStage = 2; }
      else { setSel("to", pt); pickStage = 0; document.getElementById("pick").style.background = ""; doRoute(); }
    }
  }
  cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", up);
  cv.addEventListener("wheel", function (ev) { ev.preventDefault(); zoomAt(ev.deltaY < 0 ? 1.25 : 0.8, ev.clientX, ev.clientY); }, { passive: false });
  document.getElementById("zin").onclick = function () { zoomAt(1.5, innerWidth / 2, innerHeight / 2); };
  document.getElementById("zout").onclick = function () { zoomAt(1 / 1.5, innerWidth / 2, innerHeight / 2); };
  document.getElementById("pick").onclick = function () {
    pickStage = 1; this.style.background = "#2e7d32";
    panel("<div class='meta'>Πάτα στον χάρτη την αφετηρία και μετά τον προορισμό.</div>");
  };

  // ---- αναζήτηση/επιλογή ----
  function setSel(which, item) {
    sel[which] = item;
    document.getElementById(which).value = item.name;
    dirty = true;
  }
  function wireSearch(which) {
    var inp = document.getElementById(which), box = document.getElementById("s" + which), items = [], on = -1, timer = null;
    function render() {
      box.innerHTML = items.map(function (it, i) {
        return "<div data-i='" + i + "' class='" + (i === on ? "on" : "") + "'>" + esc(it.name) + "<small>" + esc(it.type || "") + "</small></div>";
      }).join("");
      box.style.display = items.length ? "block" : "none";
    }
    inp.addEventListener("input", function () {
      sel[which] = null; clearTimeout(timer);
      timer = setTimeout(function () { if (!M) return; items = M.search(inp.value, 8); on = -1; render(); }, 150);
    });
    inp.addEventListener("keydown", function (ev) {
      if (ev.key === "ArrowDown") { on = Math.min(items.length - 1, on + 1); render(); ev.preventDefault(); }
      else if (ev.key === "ArrowUp") { on = Math.max(0, on - 1); render(); ev.preventDefault(); }
      else if (ev.key === "Enter") {
        if (items.length) { setSel(which, items[Math.max(on, 0)]); box.style.display = "none"; }
        if (sel.from && sel.to) doRoute();
      }
    });
    box.addEventListener("mousedown", function (ev) {
      var d = ev.target.closest("div[data-i]"); if (!d) return;
      ev.preventDefault(); setSel(which, items[+d.getAttribute("data-i")]); box.style.display = "none";
      var it = sel[which]; view.cx = it.lon; view.cy = it.lat; if (view.s < 20000) view.s = 20000;
    });
    inp.addEventListener("blur", function () { setTimeout(function () { box.style.display = "none"; }, 150); });
  }
  function panel(html) { document.getElementById("panel").innerHTML = html; }
  function resolve(which) {
    if (sel[which]) return sel[which];
    var r = M.search(document.getElementById(which).value, 1);
    if (r.length) setSel(which, r[0]);
    return sel[which];
  }
  function doRoute() {
    if (!M) return;
    var a = resolve("from"), b = resolve("to"), mode = document.getElementById("mode").value;
    if (!a || !b) { panel("<div class='tot'>Δεν βρέθηκε " + (!a ? "η αφετηρία" : "ο προορισμός") + "</div>"); return; }
    panel("<div class='meta'>" + (mode === "transit" && !M.graphs["transit|time"] ? "Χτίζω τον γράφο ΜΜΜ (μία φορά)…" : "Υπολογισμός…") + "</div>");
    setTimeout(function () {
      var r;
      try { r = M.route(a, b, mode); } catch (err) { panel("<div class='tot'>Σφάλμα</div><div class='meta'>" + esc(err.message) + "</div>"); throw err; }
      result = r; dirty = true;
      if (!r) { panel("<div class='tot'>Δεν βρέθηκε διαδρομή για αυτό το μέσο</div>"); return; }
      var h = "<div class='tot'>" + esc(a.name) + " → " + esc(b.name) + "<br>~" + fmtT(r.cost) + " · " + fmtD(r.dist) + " οδικά</div>";
      // συνεχόμενα βήματα περπατήματος σε ένα
      var st2 = [];
      r.steps.forEach(function (s) {
        var l = st2[st2.length - 1];
        if (s.type === "walk" && l && l.type === "walk") { l.dist += s.dist; l.time += s.time; if (l.name.indexOf(s.name) < 0 && /[Α-Ωα-ωA-Za-z]{3}/.test(s.name) && !/_|way$|^steps$|^path$|^service$/.test(s.name)) l.name += ", " + s.name; }
        else st2.push(s.type === "walk" ? { type: "walk", name: /_|way$|^steps$|^path$|^service$/.test(s.name) ? "Περπάτημα" : "Περπάτημα: " + s.name, dist: s.dist, time: s.time } : s);
      });
      st2.forEach(function (s) {
        if (s.type === "ride") h += "<div class='st'>🚌 <b>" + esc(s.name) + "</b><br>" + esc(M.poiName(s.from)) + " → " + esc(M.poiName(s.to)) + ", " + s.stops + " στάσεις, ~" + fmtT(s.time) + " (με αναμονή)</div>";
        else h += "<div class='st'>" + ({ walk: "🚶", car: "🚗", bike: "🚲" }[s.type]) + " " + esc(s.name) + " · " + fmtD(s.dist) + " · ~" + fmtT(s.time) + "</div>";
      });
      if (r.snap[0] > 300 || r.snap[1] > 300) h += "<div class='meta'>! απόσταση από δίκτυο " + fmtD(r.snap[0]) + " / " + fmtD(r.snap[1]) + "</div>";
      h += "<div class='meta'>Dijkstra: " + r.settled.toLocaleString("el") + " κορυφές σε " + Math.round(r.ms) + " ms" +
        (r.graphMs > 50 ? " · γράφος " + Math.round(r.graphMs) + " ms" : "") + memInfo() + "</div>";
      panel(h);
      // κεντράρισμα
      var lo = [], la = [];
      r.segs.forEach(function (s) { lo = lo.concat(s.lon); la = la.concat(s.lat); });
      if (lo.length) fit(Math.min.apply(null, lo), Math.min.apply(null, la), Math.max.apply(null, lo), Math.max.apply(null, la));
    }, 30);
  }
  function fit(x0, y0, x1, y1) {
    view.cx = (x0 + x1) / 2; view.cy = (y0 + y1) / 2;
    var c = Math.cos(view.cy * D2R);
    view.s = Math.max(150, Math.min(200000, 0.8 * Math.min(innerWidth / Math.max((x1 - x0) * c, 1e-4), (innerHeight - 160) / Math.max(y1 - y0, 1e-4))));
    dirty = true;
  }
  function memInfo() {
    var m = performance.memory;
    return m ? " · JS heap " + Math.round(m.usedJSHeapSize / 1e6) + " MB" : "";
  }
  document.getElementById("go").onclick = doRoute;
  document.getElementById("mode").onchange = function () { if (sel.from && sel.to) doRoute(); };
  wireSearch("from"); wireSearch("to");

  // ---- φόρτωση ----
  function load() {
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "attica.bin");
    xhr.responseType = "arraybuffer";
    xhr.onprogress = function (ev) { setLoad("Λήψη χάρτη… " + (ev.loaded / 1e6).toFixed(1) + " MB", Math.min(ev.loaded / 21e6, 0.95)); };
    xhr.onerror = function () { setLoad("Αποτυχία λήψης attica.bin"); };
    xhr.onload = function () {
      if (xhr.status !== 200) { setLoad("Σφάλμα " + xhr.status + " στη λήψη attica.bin"); return; }
      setLoad("Αποκωδικοποίηση…", 0.97);
      setTimeout(function () {
        try {
          var t0 = performance.now();
          M = new AtticaMap(parse(xhr.response));
          prepStyles();
          var ms = performance.now() - t0, c = M.meta.counts;
          document.getElementById("load").style.display = "none";
          document.getElementById("info").innerHTML = esc(M.meta.name) + " · " + c.vertices.toLocaleString("el") + " κόμβοι · " +
            c.pois.toLocaleString("el") + " POI · " + c.routes + " γραμμές<br>φόρτωση " + Math.round(ms) + " ms" + memInfo() +
            "<br>© OpenStreetMap contributors · Wikivoyage";
          window.ATTICA = M;   // για δοκιμές από την κονσόλα
          dirty = true;
        } catch (err) { setLoad("Σφάλμα: " + err.message); throw err; }
      }, 20);
    };
    xhr.send();
  }
  window.addEventListener("resize", resize);
  resize(); loop(); load();
})();
