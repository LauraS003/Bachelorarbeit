# 4. Schlechtwegfahrt

Dieser Ordner enthält die Auswertung einer real aufgezeichneten Schlechtwegfahrt.

Im Gegensatz zu den Laborversuchen handelt es sich hierbei um eine reale Fahrmessung, bei der die Kamera im Fahrzeug verbaut war und echten fahrdynamischen Anregungen ausgesetzt wurde.

Ziel ist die Analyse des zeitlichen Schärfeverlaufs unter realen Straßenbedingungen sowie die Identifikation von Fahrsituationen mit erhöhtem Schärfeverlust (z. B. Kurvenfahrt).

---

## Dateien und Funktion

---

### `schlechtwegfahrt_schaerfe.py`

Extrahiert frameweise Schärfemetriken direkt aus dem Video.

**Funktion:**

- Einlesen eines MP4-Videos
- Optionales ROI-Cropping
- Umwandlung in Graubild
- Berechnung je Frame:
  - Tenengrad
  - Laplacian-Varianz
- Speicherung der Zeitreihe als CSV
- Erstellung eines Gesamtplots (Schärfe über Zeit)

**Output:**

- `*_sharpness.csv`
- `*_sharpness.png`

**Zweck:**

Erzeugung einer vollständigen Schärfe-Zeitreihe der realen Schlechtwegfahrt als Datengrundlage für weitere Analysen.

---

### `schelchtweg_ba_plots.py`

Weiterführende Analyse und Visualisierung der zuvor berechneten Schärfe-Zeitreihe.  
(Verwendet die erzeugte CSV-Datei.)

**Funktion:**

- Einlesen der Schärfe-Zeitreihe (CSV)
- Optional:
  - Downsampling
  - Glättung (Moving Average)
- Darstellung:
  - Gesamtverlauf
  - Zoom-Ausschnitt
  - Separater Kurvenausschnitt
- Berechnung robuster Kennwerte für definierte Zeitfenster:
  - Median
  - P05 / P95
  - Range

**Output:**

- Gesamtplot
- Zoomplot
- Kurvenausschnitt-Plot
- Bereinigte Zeitreihe (CSV)

**Zweck:**

Identifikation von Fahrsituationen mit erhöhtem Schärfeverlust  
(z. B. Kurvenfahrt, Unebenheiten).

---

