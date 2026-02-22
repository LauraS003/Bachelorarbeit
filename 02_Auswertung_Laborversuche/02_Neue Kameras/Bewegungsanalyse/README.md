# Bewegungsanalyse – Optical Flow (Neue Kameras)

Dieser Ordner enthält Skripte zur Analyse der bildseitigen Relativbewegung mithilfe des optischen Flusses (Farnebäck-Verfahren).

Ziel ist die Quantifizierung der vibrationsinduzierten Pixelverschiebung zwischen aufeinanderfolgenden Frames sowie die Verknüpfung mit gemessenen Schärfemetriken.

---

## 1. optical_flow.py

Zweck:
- Berechnung eines kompakten Bewegungskennwerts pro Video
- Aggregation der Optical-Flow-Magnituden über Frequenzen
- Optionaler Vergleich mit einer Schärfemetrik (z. B. Laplace)

Funktionsweise:
- Einlesen aller Videos je Frequenzordner
- Berechnung des dichten optischen Flusses (Farnebäck)
- Ermittlung robuster Kennwerte:
  - Median |Flow| [px/Frame]
  - 90%-Quantil
  - RMS-Wert
- Erstellung eines Summary-CSV
- Normierte Darstellung: Flow vs. Frequenz
- Optionales Overlay mit Schärfemetrik
- Korrelation zwischen Bewegung und Schärfe

Ergebnis:
- Frequenzabhängiger Bewegungsverlauf
- Visualisierung des Zusammenhangs „mehr Bewegung → geringere Schärfe“

---

## 2. optical_flow_pro_vid.py

Zweck:
- Zeitliche Analyse des optischen Flusses pro Video

Funktionsweise:
- Frameweiser Optical-Flow zwischen aufeinanderfolgenden Bildern
- Optionale ROI-Nutzung (zentraler Bildbereich)
- Speicherung einer Zeitreihe:
  - Median |Flow|
  - Mean |Flow|
  - 95%-Quantil
- Erstellung von:
  - CSV mit Zeitverlauf
  - Plot „Optical Flow über Zeit“

Ergebnis:
- Sichtbarmachung dynamischer Bewegungsverläufe
- Identifikation instationärer Abschnitte
- Vergleich unterschiedlicher Anregungsformen

---

## Abhängigkeiten

- OpenCV (cv2)
- NumPy
- Pandas
- Matplotlib

Installation z. B.:
pip install opencv-python numpy pandas matplotlib