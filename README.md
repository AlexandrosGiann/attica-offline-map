# Χάρτης Αττικής offline + γεωεντοπισμός φωτογραφιών

Χάρτης της Αττικής χωρίς internet (δρόμοι, διαδρομές με Dijkstra, στάσεις ΜΜΜ, σημεία Wikivoyage) και εργαλείο που βρίσκει πού τραβήχτηκε μια φωτογραφία. Βγάζει τα 5 πιο πιθανά σημεία μαζί με τις εικόνες τους.

## Αρχεία

| Αρχείο | Τι είναι |
|---|---|
| `attica.amap` (9,9 MB) | Χάρτης: γράφος δρόμων, όρια, στάσεις, POI (xz) |
| `attica.py`, `amap.py` | Εφαρμογή γραμμής εντολών και αναγνώστης .amap |
| `build.py`, `extract.py`, `wikivoyage.py` | Κατασκευή του .amap από OSM PBF και Wikivoyage ZIM |
| `web/`, `server.py`, `web_export.py` | Έκδοση για browser στο τοπικό δίκτυο |
| `attica_geo.pack` (27,8 MB) | 2.076 εικόνες Wikidata από 1.940 σημεία (WebP 320 px) με embeddings MobileCLIP int8 και OCR |
| `geo/` | `attica-geo` σε Rust: αναζήτηση φωτογραφίας → top‑5 συντεταγμένες |
| `wd_fetch.py`, `geo_pack.py`, `geo_testset.py`, `geo_eval.py` | Λήψη, πακετάρισμα και αξιολόγηση |
| `attica_demo.mp4` | Βίντεο επίδειξης |

## Γεωεντοπισμός

Το `attica_geo.pack` είναι 27,8 MB. Πλήρως αποσυμπιεσμένο θα έπιανε περίπου 462 MB, γι' αυτό η αναζήτηση διαβάζει από το αρχείο μόνο τις εικόνες των αποτελεσμάτων.

Η αναζήτηση γίνεται στα εξής βήματα:
1. Το MobileCLIP-S0 υπολογίζει embeddings σε 1 ή 3 τετράγωνα κομμάτια της φωτογραφίας.
2. Το Mirage εντοπίζει ανθρώπους και κείμενο (OCR). Όταν οι άνθρωποι καλύπτουν πάνω από 15% της εικόνας, τους θολώνει και δοκιμάζει κομμάτια χωρίς ανθρώπους.
3. Οι 30 καλύτερες υποψήφιες ελέγχονται γεωμετρικά με FAST/BRIEF και RANSAC homography.
4. Όσες έχουν κείμενο ίδιο με το OCR της φωτογραφίας παίρνουν μπόνους.
5. Αν η φωτογραφία έχει GPS στα EXIF, η απόσταση από αυτό αναφέρεται σε κάθε αποτέλεσμα.

**Ακρίβεια** σε 30 φωτογραφίες που δεν είναι στο πακέτο:

| Αποτέλεσμα | Φωτογραφίες |
|---|---|
| Σωστό σημείο 1ο | 7/30 |
| Σωστό σημείο στα 5 πρώτα | 13/30 |
| Κάποιο από τα 5 σε απόσταση έως 1 km | 20/30 |

Κάθε αναζήτηση θέλει περίπου 4,5 s σε CPU.

### Build

Χρειάζονται δίπλα στο project:
- `ext/mc`: clone του [MobileCLIP-S0_PureRust](https://github.com/AlexandrosGiann/MobileCLIP-S0_PureRust)
- `ext/mirage.cact`: το μοντέλο Mirage
- `../Cactus Mirage - PureRust`: ο κώδικας του Mirage σε Rust

```
cd geo && cargo build --release
geo/target/release/attica-geo query photo.jpg --out apotelesma
```

### Web

```
python web_export.py      # φτιάχνει το web/attica.bin
python server.py 8080
```
Ο χάρτης ανοίγει στο `http://<ip>:8080/` και ο γεωεντοπισμός στο `http://<ip>:8080/geo.html`.

## Άδειες δεδομένων

- **Χάρτης:** © συντελεστές του OpenStreetMap, ODbL.
- **Εικόνες στο `attica_geo.pack`:** από το Wikimedia Commons, με τις άδειες των δημιουργών τους (συνήθως CC BY / CC BY-SA). Το όνομα του αρχείου Commons κάθε εικόνας είναι αποθηκευμένο στο meta του pack, μαζί με το Q-id του Wikidata.
- **Σημεία:** Wikidata (CC0) και Wikivoyage (CC BY-SA).
