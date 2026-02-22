# 3. Gealterte Kameras

Dieser Ordner enthält die Auswertung der Laboraufnahmen gealterter Kamerasysteme sowie den Vergleich mit neuwertigen Referenzkameras.

Ziel ist die Untersuchung möglicher Alterungseffekte auf das vibrationsabhängige Schärfeverhalten.

---

## Dateien und Funktion

### `Tenengrad_alt.py`

Berechnet die Tenengrad-Schärfemetrik für gealterte Kameras.

**Funktion:**
- ROI-basiertes Cropping (zentraler Bildbereich)
- Optional: Grünkanal-Verwendung zur Reduktion von Demosaicing-Artefakten
- Optional: leichte Vorfilterung (Gaussian Blur)
- Frameweise Tenengrad-Berechnung
- Statistische Kennwerte (Mean, Median, IQR, etc.)
- Frequenzabhängige Diagramme
- Normierte Darstellungen
- Diagnoseplots (Mean vs. Median)
- Optional: Violinplots und Zeitreihenplots

**Zweck:**
Analyse des Schärfeverlaufs gealterter Kameras über verschiedene Anregungsfrequenzen.

---

### `Laplace_alt.py`

Auswertung der Laplace-Varianz für gealterte Kameras mit Vergleich zur Baseline.

**Funktion:**
- Berechnung der Laplace-Varianz je Frame
- ROI-basierte Auswertung
- Grünkanal + optionale Glättung
- Statistische Kennwerte (Mean, Median, IQR, etc.)
- Ruhe-Normalisierung (Wert / Ruhe)
- Vergleich Baseline vs. gealterte Kameras
- Absolute und relative Frequenzplots
- Kamera/Baseline-Ratio-Darstellungen
- Pivot-Tabellen (CSV & optional Excel)

**Zweck:**
Analyse des Schärfeverlaufs gealterter Kameras über verschiedene Anregungsfrequenzen.

---

### `Alt_Einzeln.py`

Erstellt pro gealterter Kamera separate Frequenzverläufe.

**Funktion:**
- Einzelauswertung je Kamera
- Sinus- und Rechteckvergleich
- Absolute und relativ-zu-Ruhe-Darstellungen
- Plot-Export pro Kamera

**Zweck:**
Individuelle Charakterisierung jeder gealterten Kamera.

---

### `alt_v_neu.py`

Vergleichsauswertung zwischen gealterten und neuwertigen Kameras.

**Funktion:**
- Einlesen beider Datensätze (Alt & Neu)
- Automatische Kamerazuweisung
- Frequenzweise relative Normierung
- Robustheitskennwerte:
  - delta_p95_minus_p05 (robuste Range)
  - delta_max_minus_min
  - Mittelwert- und Median-AUC
- Gruppenstatistik (Alt vs. Neu)
- CSV-Export der Vergleichstabellen

**Zweck:**
Quantitative Bewertung möglicher Alterungseffekte über das gesamte Frequenzspektrum.

Wichtige Kennzahl für den Alterungsvergleich:
`delta_p95_minus_p05` (robuste Frequenz-Range)

---

