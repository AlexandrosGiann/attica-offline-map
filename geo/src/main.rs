//! attica-geo: σε ποιο σημείο της Αττικής τραβήχτηκε μια φωτογραφία.
//!
//! Χαρακτηριστικά ανά εικόνα του Wikidata:
//!  - MobileCLIP-S0 embeddings (512) σε 1 ή 3 τετράγωνα κομμάτια (αριστερά/κέντρο/δεξιά),
//!    ώστε οι πανοραμικές φωτογραφίες να μη χάνουν τις άκρες τους. Αποθήκευση int8.
//!  - Mirage OCR (λατινικό κείμενο: πινακίδες, επιγραφές) για ταίριασμα με ονόματα.
//!  - Ετικέτες σκηνής από το text bank του MobileCLIP (για προβολή).
//! Ερώτημα: EXIF GPS αν υπάρχει, embeddings 4 κομματιών (+ καθρέφτης), Mirage σκηνή/αντικείμενα/OCR,
//! βαθμός = μέγιστο cosine + μπόνους OCR, ομαδοποίηση σε σημεία (100 m), top-5.
//!
//! Εντολές:
//!   attica-geo query <φωτο> [--pack P] [--out DIR] [--top 5] [--no-mirage]
//!   attica-geo info [--pack P]
//!   attica-geo extract <DIR> [--pack P] [αριθμοί εικόνων...]   (μερική ή πλήρης αποσυμπίεση)
//!   attica-geo embed <manifest.json> <rawdir> <featdir>          (χτίσιμο)
//!   attica-geo ocr <featdir> <rawdir>                            (χτίσιμο, συνεχίζει από όπου σταμάτησε)

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use image::imageops::{self, FilterType};
use image::{DynamicImage, RgbImage};
use mobileclip_s0::{preprocess, MobileClip};
use serde_json::{json, Value};

type R<T> = Result<T, Box<dyn Error>>;
const DIM: usize = 512;
const MAGIC: &[u8] = b"AGEO1\n";

// ---------------------------------------------------------------- εύρεση αρχείων

fn search_up(name: &str) -> Option<PathBuf> {
    let mut starts = vec![std::env::current_dir().ok()?];
    if let Ok(e) = std::env::current_exe() {
        starts.push(e.parent()?.to_path_buf());
    }
    for s in starts {
        let mut d = Some(s.as_path());
        while let Some(p) = d {
            let c = p.join(name);
            if c.exists() {
                return Some(c);
            }
            d = p.parent();
        }
    }
    None
}

fn clip_root(args: &Args) -> R<PathBuf> {
    if let Some(p) = args.opt("--clip") {
        return Ok(p.into());
    }
    search_up("ext/mc/compiled-fp32-v1").and_then(|p| p.parent().map(|x| x.to_path_buf()))
        .ok_or_else(|| "δεν βρέθηκε το ext/mc (MobileCLIP)· δώσε --clip".into())
}

fn mirage_path(args: &Args) -> R<PathBuf> {
    if let Some(p) = args.opt("--mirage") {
        return Ok(p.into());
    }
    search_up("ext/mirage.cact").or_else(mirage::find_weights)
        .ok_or_else(|| "δεν βρέθηκε το mirage.cact· δώσε --mirage".into())
}

// ---------------------------------------------------------------- ορίσματα

struct Args {
    pos: Vec<String>,
    kv: HashMap<String, String>,
    flags: HashSet<String>,
}

impl Args {
    fn parse() -> Args {
        let mut a = Args { pos: vec![], kv: HashMap::new(), flags: HashSet::new() };
        let v: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < v.len() {
            let s = &v[i];
            if ["--pack", "--out", "--top", "--clip", "--mirage", "--threads"].contains(&s.as_str()) && i + 1 < v.len() {
                a.kv.insert(s.clone(), v[i + 1].clone());
                i += 2;
            } else if s.starts_with("--") {
                a.flags.insert(s.clone());
                i += 1;
            } else {
                a.pos.push(s.clone());
                i += 1;
            }
        }
        a
    }
    fn opt(&self, k: &str) -> Option<&str> {
        self.kv.get(k).map(|s| s.as_str())
    }
}

// ---------------------------------------------------------------- εικόνες -> embeddings

fn square(img: &RgbImage, x: u32, y: u32, side: u32) -> Vec<f32> {
    let c = imageops::crop_imm(img, x, y, side, side).to_image();
    let s = mobileclip_s0::IMAGE_SIZE as u32;
    preprocess::to_nchw(&imageops::resize(&c, s, s, FilterType::Triangle))
}

/// 1 τετράγωνο αν η εικόνα είναι σχεδόν τετράγωνη, αλλιώς 3 κατά μήκος της μεγάλης πλευράς.
/// `mirror`: και καθρέφτης του κεντρικού (για το ερώτημα).
fn crops(img: &DynamicImage, mirror: bool) -> Vec<Vec<f32>> {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let side = w.min(h);
    let mut out = vec![square(&rgb, (w - side) / 2, (h - side) / 2, side)];
    if w.max(h) as f32 >= 1.25 * side as f32 {
        if w > h {
            out.push(square(&rgb, 0, 0, side));
            out.push(square(&rgb, w - side, 0, side));
        } else {
            out.push(square(&rgb, 0, 0, side));
            out.push(square(&rgb, 0, h - side, side));
        }
    }
    if mirror {
        let f = imageops::flip_horizontal(&rgb);
        out.push(square(&f, (w - side) / 2, (h - side) / 2, side));
    }
    out
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= n);
}

fn embed_all(model: &MobileClip, img: &DynamicImage, mirror: bool) -> R<Vec<Vec<f32>>> {
    let mut out = vec![];
    for c in crops(img, mirror) {
        let mut e = model.embed(&c)?;
        normalize(&mut e);
        out.push(e);
    }
    Ok(out)
}

fn cmd_embed(args: &Args) -> R<()> {
    let (man, raw, outd) = (&args.pos[1], Path::new(&args.pos[2]), Path::new(&args.pos[3]));
    let items: Value = serde_json::from_slice(&fs::read(man)?)?;
    let mut seen = HashSet::new();
    let mut hashes = vec![];
    for it in items.as_array().ok_or("manifest")? {
        let h = it["img"].as_str().ok_or("img")?.to_string();
        if seen.insert(h.clone()) {
            hashes.push(h);
        }
    }
    let threads = args.opt("--threads").map(|s| s.parse()).transpose()?.unwrap_or(0);
    let model = MobileClip::load(&clip_root(args)?, threads)?;
    fs::create_dir_all(outd)?;
    let mut emb = std::io::BufWriter::new(File::create(outd.join("emb.f32"))?);
    let (mut cptr, mut dims) = (vec![0u32], vec![]);
    let t = Instant::now();
    for (i, h) in hashes.iter().enumerate() {
        let img = preprocess::decode(&raw.join(format!("{h}.jpg")))?;
        dims.push(json!([img.width(), img.height()]));
        let es = embed_all(&model, &img, false)?;
        for e in &es {
            for x in e {
                emb.write_all(&x.to_le_bytes())?;
            }
        }
        cptr.push(cptr.last().unwrap() + es.len() as u32);
        if (i + 1) % 100 == 0 {
            eprintln!("  {}/{}  {:.0} s", i + 1, hashes.len(), t.elapsed().as_secs_f32());
        }
    }
    emb.flush()?;
    fs::write(outd.join("images.json"), serde_json::to_vec(&json!({"img": hashes, "dims": dims, "cptr": cptr}))?)?;
    eprintln!("embeddings: {} εικόνες, {} κομμάτια, {:.0} s", hashes.len(), cptr.last().unwrap(), t.elapsed().as_secs_f32());
    Ok(())
}

fn to_mirage(img: &DynamicImage) -> R<mirage::image::Rgb> {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    Ok(mirage::image::Rgb::new(w as usize, h as usize, rgb.into_raw())?)
}

fn cmd_ocr(args: &Args) -> R<()> {
    let (featd, raw) = (Path::new(&args.pos[1]), Path::new(&args.pos[2]));
    let idx: Value = serde_json::from_slice(&fs::read(featd.join("images.json"))?)?;
    let path = featd.join("ocr.jsonl");
    let mut done = HashSet::new();
    if let Ok(f) = File::open(&path) {
        for l in BufReader::new(f).lines() {
            if let Ok(v) = serde_json::from_str::<Value>(&l?) {
                done.insert(v["img"].as_str().unwrap_or("").to_string());
            }
        }
    }
    let m = mirage::Mirage::open(mirage_path(args)?)?;
    let opts = mirage::Options::from_json(&json!({"sections": ["text"], "text_conf": 0.4}))?;
    let mut out = fs::OpenOptions::new().create(true).append(true).open(&path)?;
    let t = Instant::now();
    let all = idx["img"].as_array().ok_or("images.json")?;
    let mut n = 0;
    for h in all.iter().filter_map(|v| v.as_str()) {
        if done.contains(h) {
            continue;
        }
        let img = preprocess::decode(&raw.join(format!("{h}.jpg")))?;
        let r = m.describe(&to_mirage(&img)?, &opts, &mut |_| {})?;
        let txt: Vec<Value> = r.result["text"].as_array().cloned().unwrap_or_default().into_iter()
            .map(|t| json!([t["string"], t["score"]])).collect();
        writeln!(out, "{}", json!({"img": h, "text": txt}))?;
        n += 1;
        if n % 50 == 0 {
            eprintln!("  OCR {}/{}  {:.0} s", n + done.len(), all.len(), t.elapsed().as_secs_f32());
        }
    }
    eprintln!("OCR: {} νέες εικόνες σε {:.0} s", n, t.elapsed().as_secs_f32());
    Ok(())
}

// ---------------------------------------------------------------- πακέτο

struct Pack {
    file: File,
    sec: HashMap<String, (u64, u64)>,
    meta: Value,
    emb: Vec<i8>,
    scale: Vec<f32>,
    cptr: Vec<u32>,
    iptr: Vec<u64>,
    path: PathBuf,
}

impl Pack {
    fn open(path: &Path) -> R<Pack> {
        let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut m = [0u8; 6];
        file.read_exact(&mut m)?;
        if m != MAGIC {
            return Err("δεν είναι πακέτο attica-geo".into());
        }
        let mut l = [0u8; 4];
        file.read_exact(&mut l)?;
        let mut hb = vec![0u8; u32::from_le_bytes(l) as usize];
        file.read_exact(&mut hb)?;
        let hdr: Value = serde_json::from_slice(&hb)?;
        let mut sec = HashMap::new();
        for (k, v) in hdr["sections"].as_object().ok_or("header")? {
            sec.insert(k.clone(), (v[0].as_u64().ok_or("off")?, v[1].as_u64().ok_or("len")?));
        }
        let mut p = Pack { file, sec, meta: Value::Null, emb: vec![], scale: vec![], cptr: vec![], iptr: vec![], path: path.into() };
        let xz = p.read("meta")?;
        let mut js = vec![];
        lzma_rs::xz_decompress(&mut &xz[..], &mut js).map_err(|e| format!("meta xz: {e:?}"))?;
        p.meta = serde_json::from_slice(&js)?;
        p.emb = p.read("emb")?.into_iter().map(|b| b as i8).collect();
        p.scale = p.read("scale")?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        p.cptr = p.read("cptr")?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        p.iptr = p.read("iptr")?.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        Ok(p)
    }

    fn read(&mut self, name: &str) -> R<Vec<u8>> {
        let (o, l) = *self.sec.get(name).ok_or_else(|| format!("λείπει η ενότητα {name}"))?;
        let mut b = vec![0u8; l as usize];
        self.file.seek(SeekFrom::Start(o))?;
        self.file.read_exact(&mut b)?;
        Ok(b)
    }

    /// Μόνο τα bytes μιας εικόνας (WebP): μερική ανάγνωση.
    fn image(&mut self, i: usize) -> R<Vec<u8>> {
        let (o, _) = self.sec["img"];
        let (a, b) = (self.iptr[i], self.iptr[i + 1]);
        let mut buf = vec![0u8; (b - a) as usize];
        self.file.seek(SeekFrom::Start(o + a))?;
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn n_img(&self) -> usize {
        self.cptr.len() - 1
    }
}

fn find_pack(args: &Args) -> R<PathBuf> {
    if let Some(p) = args.opt("--pack") {
        return Ok(p.into());
    }
    search_up("attica_geo.pack").ok_or_else(|| "δεν βρέθηκε το attica_geo.pack· δώσε --pack".into())
}

// ---------------------------------------------------------------- κείμενο

fn fold(c: char) -> char {
    // ελληνικά -> λατινικά «που μοιάζουν» (όπως τα διαβάζει ένα λατινικό OCR) και χωρίς τόνους
    match c {
        'ά' | 'α' => 'a', 'β' => 'b', 'γ' => 'g', 'δ' => 'd', 'έ' | 'ε' => 'e', 'ζ' => 'z',
        'ή' | 'η' => 'h', 'θ' => 'o', 'ί' | 'ϊ' | 'ΐ' | 'ι' => 'i', 'κ' => 'k', 'λ' => 'l',
        'μ' => 'm', 'ν' => 'n', 'ξ' => 'x', 'ό' | 'ο' => 'o', 'π' => 'n', 'ρ' => 'p',
        'σ' | 'ς' => 's', 'τ' => 't', 'ύ' | 'ϋ' | 'ΰ' | 'υ' => 'y', 'φ' => 'f', 'χ' => 'x',
        'ψ' => 'y', 'ώ' | 'ω' => 'w', 'à' | 'á' | 'â' | 'ä' => 'a', 'é' | 'è' | 'ê' => 'e',
        'í' | 'ï' => 'i', 'ó' | 'ö' | 'ô' => 'o', 'ú' | 'ü' => 'u', c => c,
    }
}

fn tokens(s: &str) -> Vec<String> {
    let low: String = s.to_lowercase().chars().map(fold).collect();
    low.split(|c: char| !c.is_ascii_alphanumeric()).filter(|t| t.len() >= 4).map(|t| t.to_string()).collect()
}

fn close(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    if a.len().min(b.len()) < 6 || a.len().abs_diff(b.len()) > 1 {
        return false;
    }
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + (a[i - 1] != b[j - 1]) as usize);
        }
        prev = cur;
    }
    prev[b.len()] <= 1
}

// ---------------------------------------------------------------- EXIF

fn exif_gps(path: &Path) -> Option<(f64, f64)> {
    let f = File::open(path).ok()?;
    let ex = exif::Reader::new().read_from_container(&mut BufReader::new(f)).ok()?;
    let deg = |tag, rtag| -> Option<f64> {
        let f = ex.get_field(tag, exif::In::PRIMARY)?;
        let v = match &f.value {
            exif::Value::Rational(v) if v.len() >= 3 => v[0].to_f64() + v[1].to_f64() / 60.0 + v[2].to_f64() / 3600.0,
            _ => return None,
        };
        let r = ex.get_field(rtag, exif::In::PRIMARY).map(|r| r.display_value().to_string()).unwrap_or_default();
        Some(if r.starts_with('S') || r.starts_with('W') { -v } else { v })
    };
    Some((deg(exif::Tag::GPSLatitude, exif::Tag::GPSLatitudeRef)?, deg(exif::Tag::GPSLongitude, exif::Tag::GPSLongitudeRef)?))
}

fn haversine(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let a = ((p2 - p1) / 2.0).sin().powi(2) + p1.cos() * p2.cos() * ((lon2 - lon1).to_radians() / 2.0).sin().powi(2);
    12_742_000.0 * a.sqrt().asin()
}

// ---------------------------------------------------------------- ερώτημα

fn s(v: &Value) -> &str {
    v.as_str().unwrap_or("")
}

fn cmd_query(args: &Args) -> R<()> {
    let photo = PathBuf::from(args.pos.get(1).ok_or("δώσε φωτογραφία")?);
    let top: usize = args.opt("--top").map(|s| s.parse()).transpose()?.unwrap_or(5);
    let t0 = Instant::now();
    let mut pack = Pack::open(&find_pack(args)?)?;
    let t_pack = t0.elapsed();
    let img0 = preprocess::decode(&photo)?;
    // όριο 1024 px: αρκεί για CLIP (256), Mirage (512) και τα σημεία-κλειδιά
    let img = if img0.width().max(img0.height()) > 1024 { img0.resize(1024, 1024, FilterType::Triangle) } else { img0 };
    let (w, h) = (img.width(), img.height());

    // Mirage: σκηνή, αντικείμενα με μάσκες (για τους ανθρώπους), OCR
    let mut mir = Value::Null;
    let mut ocr_tokens: Vec<String> = vec![];
    let mut people: Vec<Vec<(f32, f32)>> = vec![];
    let t2 = Instant::now();
    if !args.flags.contains("--no-mirage") {
        match mirage_path(args).and_then(|p| Ok(mirage::Mirage::open(p)?)) {
            Ok(m) => {
                let o = mirage::Options::from_json(&json!({"sections": ["scene", "objects", "text"], "max_tokens": 320,
                                                          "fields": ["bbox", "score", "position", "mask"],
                                                          "mask_format": "polygons"}))?;
                let r = m.describe(&to_mirage(&img)?, &o, &mut |_| {})?;
                for t in r.result["text"].as_array().into_iter().flatten() {
                    if t["score"].as_f64().unwrap_or(0.0) >= 0.5 {
                        ocr_tokens.extend(tokens(s(&t["string"])));
                    }
                }
                for ob in r.result["objects"].as_array().into_iter().flatten() {
                    if is_person(s(&ob["name"])) {
                        people.extend(object_polys(ob, w, h));
                    }
                }
                mir = r.result;
            }
            Err(e) => eprintln!("Mirage παραλείπεται: {e}"),
        }
    }
    let t_mir = t2.elapsed();

    // παραλλαγές: αρχική (+καθρέφτης), άνθρωποι θολωμένοι, τετράγωνα χωρίς ανθρώπους
    let mask = person_mask(&people, w, h);
    let cover = mask.frac();
    let model = MobileClip::load(&clip_root(args)?, 0)?;
    let t1 = Instant::now();
    let mut q = embed_all(&model, &img, true)?;
    let mut variants = vec![format!("αρχική: {} κομμάτια", q.len())];
    // μετρήθηκε: με λίγους ανθρώπους (<15%) οι επιπλέον παραλλαγές φέρνουν περισσότερα λάθος ταιριάσματα
    if cover > 0.15 {
        let bl = blur_people(&img, &mask);
        let e = embed_all(&model, &bl, false)?;
        variants.push(format!("θολωμένοι άνθρωποι: {}", e.len()));
        q.extend(e);
        let rgb = img.to_rgb8();
        let wins = clean_windows(&mask, w, h);
        for &(x, y, sd) in &wins {
            let mut e = model.embed(&square(&rgb, x, y, sd))?;
            normalize(&mut e);
            q.push(e);
        }
        variants.push(format!("τετράγωνα χωρίς ανθρώπους: {}", wins.len()));
    }
    let t_clip = t1.elapsed();

    // ομοιότητα ανά εικόνα: μέγιστο cosine ανάμεσα σε κομμάτια ερωτήματος και εικόνας
    let t3 = Instant::now();
    let n = pack.n_img();
    let mut sim = vec![f32::MIN; n];
    for i in 0..n {
        for c in pack.cptr[i] as usize..pack.cptr[i + 1] as usize {
            let row = &pack.emb[c * DIM..(c + 1) * DIM];
            let sc = pack.scale[c];
            for qe in &q {
                let d: f32 = row.iter().zip(qe).map(|(&a, &b)| a as f32 * b).sum::<f32>() * sc;
                if d > sim[i] {
                    sim[i] = d;
                }
            }
        }
    }
    let clip_sim = sim.clone();
    // επαλήθευση γεωμετρίας: FAST + BRIEF + homography (RANSAC) στις 30 καλύτερες
    let mut inl = vec![0usize; n];
    let t4 = Instant::now();
    if !args.flags.contains("--no-verify") {
        let qf = Feats::new(&img, Some(&mask));
        let mut cand: Vec<usize> = (0..n).collect();
        cand.sort_by(|&a, &b| sim[b].total_cmp(&sim[a]));
        for &i in cand.iter().take(30) {
            if let Ok(g) = image::load_from_memory(&pack.image(i)?) {
                inl[i] = qf.inliers(&Feats::new(&g, None));
                // μετρήθηκε σε 30 φωτογραφίες εκτός πακέτου: οι λάθος υποψήφιες φτάνουν ~20 inliers,
                // άρα μπόνους μόνο για σχεδόν ίδια λήψη
                if inl[i] > 30 {
                    sim[i] += (0.004 * (inl[i] - 30) as f32).min(0.12);
                }
            }
        }
    }
    let t_verify = t4.elapsed();
    let items = pack.meta["items"].as_array().ok_or("meta.items")?.clone();
    let images = pack.meta["images"].as_array().ok_or("meta.images")?.clone();

    // OCR: λέξεις που σπανίζουν (df <= 15) και ταιριάζουν με όνομα αντικειμένου ή OCR εικόνας
    let mut item_tok: Vec<Vec<String>> = items.iter().map(|it| {
        let mut t = tokens(&format!("{} {}", s(&it["el"]), s(&it["en"])));
        t.sort();
        t.dedup();
        t
    }).collect();
    for (i, im) in images.iter().enumerate() {
        let t = tokens(s(&im["ocr"]));
        for it in im["items"].as_array().into_iter().flatten() {
            if let Some(k) = it.as_u64() {
                item_tok[k as usize].extend(t.iter().cloned());
            }
        }
        let _ = i;
    }
    let mut df: HashMap<&str, usize> = HashMap::new();
    for t in &item_tok {
        let u: HashSet<&str> = t.iter().map(|x| x.as_str()).collect();
        for x in u {
            *df.entry(x).or_default() += 1;
        }
    }
    ocr_tokens.sort();
    ocr_tokens.dedup();
    let mut item_score = vec![f32::MIN; items.len()];
    let mut item_best = vec![usize::MAX; items.len()];
    for (i, im) in images.iter().enumerate() {
        for it in im["items"].as_array().into_iter().flatten() {
            let k = it.as_u64().unwrap_or(0) as usize;
            if sim[i] > item_score[k] {
                item_score[k] = sim[i];
                item_best[k] = i;
            }
        }
    }
    let mut ocr_hits: Vec<Vec<String>> = vec![vec![]; items.len()];
    if !ocr_tokens.is_empty() {
        for (k, toks) in item_tok.iter().enumerate() {
            let mut hit = vec![];
            for qt in &ocr_tokens {
                if let Some(t) = toks.iter().find(|t| close(qt, t)) {
                    if df.get(t.as_str()).copied().unwrap_or(0) <= 15 && !hit.contains(t) {
                        hit.push(t.clone());
                    }
                }
            }
            if !hit.is_empty() {
                item_score[k] += (0.05 * hit.len() as f32).min(0.15);
                ocr_hits[k] = hit;
            }
        }
    }

    // σημεία: ομαδοποίηση αντικειμένων σε απόσταση < 100 m
    let mut order: Vec<usize> = (0..items.len()).filter(|&k| item_best[k] != usize::MAX).collect();
    order.sort_by(|&a, &b| item_score[b].total_cmp(&item_score[a]));
    struct Pt { lat: f64, lon: f64, score: f32, items: Vec<usize> }
    let mut pts: Vec<Pt> = vec![];
    for &k in &order {
        let (lat, lon) = (items[k]["lat"].as_f64().unwrap_or(0.0), items[k]["lon"].as_f64().unwrap_or(0.0));
        if let Some(p) = pts.iter_mut().find(|p| haversine(p.lat, p.lon, lat, lon) < 100.0) {
            if p.items.len() < 4 {
                p.items.push(k);
            }
            continue;
        }
        if pts.len() >= top * 6 {
            break;
        }
        pts.push(Pt { lat, lon, score: item_score[k], items: vec![k] });
    }
    // σχετική πιθανότητα: softmax στα 30 καλύτερα σημεία
    let temp = 0.025f32;
    let mx = pts.first().map(|p| p.score).unwrap_or(0.0);
    let z: f32 = pts.iter().take(30).map(|p| ((p.score - mx) / temp).exp()).sum();
    let t_search = t3.elapsed();

    let out = PathBuf::from(args.opt("--out").unwrap_or("geo_result"));
    fs::create_dir_all(&out)?;
    let qname = format!("query.{}", photo.extension().and_then(|e| e.to_str()).unwrap_or("jpg").to_lowercase());
    fs::copy(&photo, out.join(&qname))?;
    let gps = exif_gps(&photo);
    let mut res = vec![];
    for (r, p) in pts.iter().take(top).enumerate() {
        let mut ims = vec![];
        let mut used = HashSet::new();
        for &k in &p.items {
            let i = item_best[k];
            if !used.insert(i) || ims.len() >= 3 {
                continue;
            }
            let name = format!("{}_{}.webp", r + 1, ims.len() + 1);
            fs::write(out.join(&name), pack.image(i)?)?;
            ims.push(json!({"file": name, "similarity": r3(clip_sim[i]), "inliers": inl[i],
                            "commons": images[i]["file"], "item": items[k]["q"]}));
        }
        let names: Vec<Value> = p.items.iter().map(|&k| json!({
            "q": items[k]["q"], "el": items[k]["el"], "en": items[k]["en"], "desc": items[k]["desc"],
            "score": r3(item_score[k]), "ocr_match": ocr_hits[k]})).collect();
        res.push(json!({"rank": r + 1, "lat": p.lat, "lon": p.lon,
            "score": r3(p.score),
            "probability": r3(((p.score - mx) / temp).exp() / z),
            "gps_distance_m": gps.map(|(la, lo)| haversine(la, lo, p.lat, p.lon).round()),
            "places": names, "images": ims}));
    }
    let timing = json!({"pack_load_ms": t_pack.as_millis(), "clip_ms": t_clip.as_millis(),
                        "mirage_ms": t_mir.as_millis(), "verify_ms": t_verify.as_millis(),
                        "search_ms": t_search.as_millis()});
    let low = cover > 0.6;
    let report = json!({"query": qname, "exif_gps": gps.map(|(a, b)| json!([a, b])), "mirage": mir,
                        "ocr_tokens": ocr_tokens, "results": res, "timing": timing,
                        "people_cover": r3(cover), "low_confidence": low, "variants": variants,
                        "pack": pack.path.display().to_string()});
    fs::write(out.join("result.json"), serde_json::to_vec_pretty(&report)?)?;
    fs::write(out.join("result.html"), html(&report))?;

    if let Some((la, lo)) = gps {
        println!("EXIF GPS της φωτογραφίας: {la:.6}, {lo:.6} (η πραγματική θέση· τα παρακάτω είναι από την εικόνα μόνο)");
    }
    if cover > 0.01 {
        println!("άνθρωποι: {:.0}% της εικόνας ({}){}", cover * 100.0, variants.join(", "),
                 if low { " -> ΧΑΜΗΛΗ βεβαιότητα" } else { "" });
    }
    if !ocr_tokens.is_empty() {
        println!("OCR: {}", ocr_tokens.join(" "));
    }
    for r in &res {
        let pl = &r["places"][0];
        let nm = if s(&pl["el"]).is_empty() { s(&pl["en"]) } else { s(&pl["el"]) };
        print!("{}. {:.5}, {:.5}  βαθμός {:.3}  ~{:.0}%  {} ({})", r["rank"], r["lat"].as_f64().unwrap_or(0.0),
               r["lon"].as_f64().unwrap_or(0.0), r["score"].as_f64().unwrap_or(0.0),
               r["probability"].as_f64().unwrap_or(0.0) * 100.0, nm, s(&pl["q"]));
        if let Some(d) = r["gps_distance_m"].as_f64() {
            print!("  [{:.1} km από το GPS]", d / 1000.0);
        }
        println!();
    }
    println!("χρόνοι: πακέτο {} ms, Mirage {} ms, MobileCLIP {} ms ({} κομμάτια), επαλήθευση {} ms, αναζήτηση {} ms -> {}",
             t_pack.as_millis(), t_mir.as_millis(), t_clip.as_millis(), q.len(), t_verify.as_millis(), t_search.as_millis(),
             out.join("result.html").display());
    Ok(())
}

// ---------------------------------------------------------------- άνθρωποι (segmentation)

fn is_person(name: &str) -> bool {
    const W: [&str; 22] = ["person", "people", "man", "men", "woman", "women", "boy", "girl", "child", "children", "kid",
                           "tourist", "pedestrian", "player", "crowd", "human", "lady", "guy", "worker", "visitor", "rider", "baby"];
    name.to_lowercase().split(|c: char| !c.is_alphabetic()).any(|t| W.contains(&t))
}

/// Πολύγωνα σε pixels: από τη μάσκα του Mirage (κλίμακα 0..1000 κάθε άξονα) ή, αν λείπει, από το κουτί.
fn object_polys(ob: &Value, w: u32, h: u32) -> Vec<Vec<(f32, f32)>> {
    let to = |v: &[Value]| -> Vec<(f32, f32)> {
        v.chunks_exact(2).map(|c| (c[0].as_f64().unwrap_or(0.0) as f32 * w as f32 / 1000.0,
                                   c[1].as_f64().unwrap_or(0.0) as f32 * h as f32 / 1000.0)).collect()
    };
    let mut out = vec![];
    match ob["mask"].as_array() {
        Some(a) if a.first().map_or(false, |x| x.is_array()) => {
            for p in a.iter().filter_map(|x| x.as_array()) {
                out.push(to(p));
            }
        }
        Some(a) if a.len() >= 6 => out.push(to(a)),
        _ => {
            if let Some(b) = ob["bbox"].as_array().filter(|b| b.len() == 4) {
                let p = to(b);
                out.push(vec![p[0], (p[1].0, p[0].1), p[1], (p[0].0, p[1].1)]);
            }
        }
    }
    out.retain(|p| p.len() >= 3);
    out
}

struct Mask {
    w: u32,
    h: u32,
    m: Vec<bool>,
    ii: Vec<u32>, // integral (w+1)*(h+1)
}

impl Mask {
    fn build(w: u32, h: u32, m: Vec<bool>) -> Mask {
        let (wu, hu) = (w as usize, h as usize);
        let mut ii = vec![0u32; (wu + 1) * (hu + 1)];
        for y in 0..hu {
            let mut row = 0u32;
            for x in 0..wu {
                row += m[y * wu + x] as u32;
                ii[(y + 1) * (wu + 1) + x + 1] = ii[y * (wu + 1) + x + 1] + row;
            }
        }
        Mask { w, h, m, ii }
    }
    fn count(&self, x: u32, y: u32, sw: u32, sh: u32) -> u32 {
        let (x1, y1) = ((x + sw).min(self.w) as usize, (y + sh).min(self.h) as usize);
        let (x, y, s) = (x as usize, y as usize, self.w as usize + 1);
        self.ii[y1 * s + x1] + self.ii[y * s + x] - self.ii[y * s + x1] - self.ii[y1 * s + x]
    }
    fn frac(&self) -> f32 {
        self.count(0, 0, self.w, self.h) as f32 / (self.w * self.h).max(1) as f32
    }
    fn at(&self, x: u32, y: u32) -> bool {
        self.m[(y * self.w + x) as usize]
    }
}

/// Γέμισμα πολυγώνων (even-odd) και διαστολή ~1,5% για τις άκρες που κόβει η απλοποίηση.
fn person_mask(polys: &[Vec<(f32, f32)>], w: u32, h: u32) -> Mask {
    let (wu, hu) = (w as usize, h as usize);
    let mut m = vec![false; wu * hu];
    if !polys.is_empty() {
        for y in 0..hu {
            let yc = y as f32 + 0.5;
            let mut xs = vec![];
            for p in polys {
                for k in 0..p.len() {
                    let (a, b) = (p[k], p[(k + 1) % p.len()]);
                    if (a.1 > yc) != (b.1 > yc) {
                        xs.push(a.0 + (yc - a.1) * (b.0 - a.0) / (b.1 - a.1));
                    }
                }
            }
            xs.sort_by(|a, b| a.total_cmp(b));
            for c in xs.chunks_exact(2) {
                let (x0, x1) = (c[0].max(0.0).round() as usize, (c[1].round().max(0.0) as usize).min(wu));
                for x in x0..x1 {
                    m[y * wu + x] = true;
                }
            }
        }
    }
    let raw = Mask::build(w, h, m);
    if raw.frac() == 0.0 {
        return raw;
    }
    let r = ((w.max(h) as f32) * 0.015).ceil() as u32;
    let mut d = vec![false; wu * hu];
    for y in 0..h {
        for x in 0..w {
            let (x0, y0) = (x.saturating_sub(r), y.saturating_sub(r));
            d[(y * w + x) as usize] = raw.count(x0, y0, x + r + 1 - x0, y + r + 1 - y0) > 0;
        }
    }
    Mask::build(w, h, d)
}

/// Οι άνθρωποι αντικαθίστανται από έντονο θόλωμα της ίδιας εικόνας (όχι από pixels άλλης εικόνας).
fn blur_people(img: &DynamicImage, mask: &Mask) -> DynamicImage {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let small = imageops::resize(&rgb, (w / 24).max(2), (h / 24).max(2), FilterType::Triangle);
    let big = imageops::resize(&small, w, h, FilterType::Triangle);
    let mut out = rgb.clone();
    for (x, y, p) in out.enumerate_pixels_mut() {
        if mask.at(x, y) {
            *p = *big.get_pixel(x, y);
        }
    }
    DynamicImage::ImageRgb8(out)
}

/// Έως 3 τετράγωνα (100/80/65% της μικρής πλευράς) με σχεδόν καθόλου ανθρώπους.
fn clean_windows(mask: &Mask, w: u32, h: u32) -> Vec<(u32, u32, u32)> {
    let mut c = vec![];
    for f in [1.0f32, 0.8, 0.65] {
        let sd = (w.min(h) as f32 * f) as u32;
        if sd < 96 {
            continue;
        }
        for gy in 0..5 {
            for gx in 0..5 {
                let (x, y) = ((w - sd) * gx / 4, (h - sd) * gy / 4);
                let cov = mask.count(x, y, sd, sd) as f32 / (sd * sd) as f32;
                if cov < 0.03 {
                    c.push((cov - f * 0.01, x, y, sd));
                }
            }
        }
    }
    c.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out: Vec<(u32, u32, u32)> = vec![];
    for (_, x, y, sd) in c {
        let iou = |o: &(u32, u32, u32)| {
            let ix = (x + sd).min(o.0 + o.2) as i64 - x.max(o.0) as i64;
            let iy = (y + sd).min(o.1 + o.2) as i64 - y.max(o.1) as i64;
            let i = (ix.max(0) * iy.max(0)) as f32;
            i / ((sd * sd + o.2 * o.2) as f32 - i)
        };
        if out.iter().all(|o| iou(o) < 0.5) {
            out.push((x, y, sd));
            if out.len() == 3 {
                break;
            }
        }
    }
    out
}

// ---------------------------------------------------------------- επαλήθευση γεωμετρίας

use imageproc::binary_descriptors::brief::{brief, BriefDescriptor, TestPair};
use imageproc::binary_descriptors::BinaryDescriptor;
use imageproc::geometric_transformations::Projection;
use imageproc::point::Point;

fn brief_pairs() -> &'static Vec<TestPair> {
    static P: std::sync::OnceLock<Vec<TestPair>> = std::sync::OnceLock::new();
    P.get_or_init(|| {
        // σταθερά ζεύγη (ίδια για ερώτημα και υποψήφιες): Gaussian γύρω από το κέντρο, σ≈6,6
        let mut st = 0x9E3779B97F4A7C15u64;
        let mut u = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            (st >> 11) as f32 / (1u64 << 53) as f32
        };
        let mut g = || {
            let v: f32 = (0..12).map(|_| u()).sum::<f32>() - 6.0;
            (16.0 + 6.6 * v).round()
        };
        let mut out = vec![];
        while out.len() < 256 {
            let (a, b, c, d) = (g(), g(), g(), g());
            if [a, b, c, d].iter().all(|&v| (0.0..31.0).contains(&v)) {
                out.push(TestPair { p0: Point::new(a as u32, b as u32), p1: Point::new(c as u32, d as u32) });
            }
        }
        out
    })
}

struct Feats {
    d: Vec<BriefDescriptor>,
}

impl Feats {
    /// Γκρι 384 px, FAST-9, τα 500 ισχυρότερα σημεία εκτός ανθρώπων, BRIEF 256 bit.
    fn new(img: &DynamicImage, mask: Option<&Mask>) -> Feats {
        let g = img.thumbnail(384, 384).to_luma8();
        let (w, h) = g.dimensions();
        let mut cs = imageproc::corners::corners_fast9(&g, 20);
        let m = 22;
        cs.retain(|c| c.x > m && c.y > m && c.x + m < w && c.y + m < h);
        if let Some(mk) = mask {
            let (sx, sy) = (mk.w as f32 / w as f32, mk.h as f32 / h as f32);
            cs.retain(|c| !mk.at(((c.x as f32 * sx) as u32).min(mk.w - 1), ((c.y as f32 * sy) as u32).min(mk.h - 1)));
        }
        cs.sort_by(|a, b| b.score.total_cmp(&a.score));
        cs.truncate(500);
        let pts: Vec<Point<u32>> = cs.iter().map(|c| Point::new(c.x, c.y)).collect();
        let d = brief(&g, &pts, 256, Some(brief_pairs())).map(|r| r.0).unwrap_or_default();
        Feats { d }
    }

    /// Ταιριάσματα με ratio test και RANSAC homography: πλήθος inliers (σφάλμα < 6 px).
    fn inliers(&self, o: &Feats) -> usize {
        let mut mt: Vec<((f32, f32), (f32, f32))> = vec![];
        for a in &self.d {
            let (mut b1, mut b2, mut bi) = (u32::MAX, u32::MAX, 0);
            for (j, b) in o.d.iter().enumerate() {
                let x = a.hamming_distance(b);
                if x < b1 {
                    b2 = b1;
                    b1 = x;
                    bi = j;
                } else if x < b2 {
                    b2 = x;
                }
            }
            if b1 < 64 && (b1 as f32) < 0.8 * b2 as f32 {
                let (p, q) = (a.position(), o.d[bi].position());
                mt.push(((p.x as f32, p.y as f32), (q.x as f32, q.y as f32)));
            }
        }
        if mt.len() < 8 {
            return 0;
        }
        let mut st = 0x2545F4914F6CDD1Du64 ^ mt.len() as u64;
        let mut rnd = |n: usize| {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            (st % n as u64) as usize
        };
        let mut best = 0;
        for _ in 0..400 {
            let mut k = [0usize; 4];
            for i in 0..4 {
                k[i] = rnd(mt.len());
            }
            if (0..4).any(|i| (0..i).any(|j| k[i] == k[j])) {
                continue;
            }
            let Some(p) = Projection::from_control_points([mt[k[0]].0, mt[k[1]].0, mt[k[2]].0, mt[k[3]].0],
                                                          [mt[k[0]].1, mt[k[1]].1, mt[k[2]].1, mt[k[3]].1]) else { continue };
            let c = mt.iter().filter(|(a, b)| {
                let (x, y) = p * *a;
                x.is_finite() && (x - b.0).powi(2) + (y - b.1).powi(2) < 36.0
            }).count();
            best = best.max(c);
        }
        if best >= 8 { best } else { 0 }
    }
}

fn r3(x: f32) -> f64 {
    ((x as f64) * 1000.0).round() / 1000.0
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

fn html(r: &Value) -> String {
    let mut h = String::from("<!doctype html><html lang=\"el\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Θέση φωτογραφίας</title><style>body{font:15px system-ui,sans-serif;margin:16px;background:#111;color:#eee}img{max-height:220px;max-width:100%;border-radius:6px;margin:4px}.r{border-top:1px solid #333;padding:10px 0}a{color:#6cf}small{color:#999}</style>");
    h += &format!("<h2>Ερώτημα</h2><img src=\"{}\">", esc(s(&r["query"])));
    if let Some(g) = r["exif_gps"].as_array() {
        h += &format!("<p>EXIF GPS: {}, {}</p>", g[0], g[1]);
    }
    if let Some(sc) = r["mirage"]["scene"].as_object() {
        let v: Vec<String> = sc.iter().map(|(k, v)| format!("{k}: {}", s(v))).collect();
        h += &format!("<p><small>Mirage σκηνή: {}</small></p>", esc(&v.join(", ")));
    }
    if r["people_cover"].as_f64().unwrap_or(0.0) > 0.01 {
        h += &format!("<p><small>άνθρωποι: {:.0}% της εικόνας{}</small></p>", r["people_cover"].as_f64().unwrap_or(0.0) * 100.0,
                      if r["low_confidence"].as_bool() == Some(true) { " — <b>χαμηλή βεβαιότητα</b>" } else { "" });
    }
    if let Some(t) = r["ocr_tokens"].as_array().filter(|t| !t.is_empty()) {
        let v: Vec<&str> = t.iter().map(s).collect();
        h += &format!("<p><small>OCR: {}</small></p>", esc(&v.join(" ")));
    }
    for x in r["results"].as_array().into_iter().flatten() {
        let (la, lo) = (x["lat"].as_f64().unwrap_or(0.0), x["lon"].as_f64().unwrap_or(0.0));
        h += &format!("<div class=r><b>{}. {:.5}, {:.5}</b> — βαθμός {} · ~{:.0}% · <a href=\"https://www.openstreetmap.org/?mlat={la}&mlon={lo}#map=17/{la}/{lo}\">χάρτης</a><br>",
                      x["rank"], la, lo, x["score"], x["probability"].as_f64().unwrap_or(0.0) * 100.0);
        for p in x["places"].as_array().into_iter().flatten() {
            h += &format!("<small>{} / {} — <a href=\"https://www.wikidata.org/wiki/{}\">{}</a> {}</small><br>",
                          esc(s(&p["el"])), esc(s(&p["en"])), esc(s(&p["q"])), esc(s(&p["q"])), esc(s(&p["desc"])));
        }
        for im in x["images"].as_array().into_iter().flatten() {
            h += &format!("<img src=\"{}\" title=\"{} (cos {}, inliers {})\">", esc(s(&im["file"])), esc(s(&im["commons"])), im["similarity"], im["inliers"]);
        }
        h += "</div>";
    }
    h += &format!("<p><small>χρόνοι: {}</small></p></html>", r["timing"]);
    h
}

fn cmd_info(args: &Args) -> R<()> {
    let p = find_pack(args)?;
    let pack = Pack::open(&p)?;
    let size = fs::metadata(&p)?.len();
    println!("{}: {:.2} MB, {} εικόνες, {} κομμάτια, {} αντικείμενα Wikidata", p.display(), size as f64 / 1e6,
             pack.n_img(), pack.scale.len(), pack.meta["items"].as_array().map(|a| a.len()).unwrap_or(0));
    let mut v: Vec<_> = pack.sec.iter().collect();
    v.sort_by_key(|x| x.1 .0);
    for (k, (_, l)) in v {
        println!("  {k:6} {:>10.2} MB", *l as f64 / 1e6);
    }
    if let Some(st) = pack.meta.get("stats") {
        println!("  {st}");
    }
    Ok(())
}

fn cmd_extract(args: &Args) -> R<()> {
    let out = PathBuf::from(args.pos.get(1).ok_or("δώσε φάκελο")?);
    let mut pack = Pack::open(&find_pack(args)?)?;
    fs::create_dir_all(&out)?;
    let ids: Vec<usize> = if args.pos.len() > 2 {
        args.pos[2..].iter().map(|x| x.parse()).collect::<Result<_, _>>()?
    } else {
        (0..pack.n_img()).collect()
    };
    let decode = args.flags.contains("--png");
    let (mut bytes, mut raw) = (0u64, 0u64);
    let t = Instant::now();
    for i in ids {
        let b = pack.image(i)?;
        if decode {
            let im = image::load_from_memory(&b)?;
            raw += (im.width() * im.height() * 3) as u64;
            im.to_rgb8().save(out.join(format!("{i:05}.png")))?;
        } else {
            fs::write(out.join(format!("{i:05}.webp")), &b)?;
        }
        bytes += b.len() as u64;
    }
    println!("{:.2} MB συμπιεσμένα{} σε {:.1} s", bytes as f64 / 1e6,
             if decode { format!(", {:.2} MB pixels", raw as f64 / 1e6) } else { String::new() }, t.elapsed().as_secs_f32());
    Ok(())
}

fn main() {
    let args = Args::parse();
    let r = match args.pos.first().map(|s| s.as_str()) {
        Some("query") => cmd_query(&args),
        Some("info") => cmd_info(&args),
        Some("extract") => cmd_extract(&args),
        Some("embed") if args.pos.len() >= 4 => cmd_embed(&args),
        Some("ocr") if args.pos.len() >= 3 => cmd_ocr(&args),
        _ => Err("χρήση: attica-geo query <φωτο> [--out DIR] [--top 5] [--no-mirage] | info | extract <DIR> [ids] [--png] | embed <manifest> <raw> <feat> | ocr <feat> <raw>".into()),
    };
    if let Err(e) = r {
        eprintln!("σφάλμα: {e}");
        std::process::exit(1);
    }
}
