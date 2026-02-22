# Darstellungen für Schriftliche Arbeit

Dieser Ordner enthält die final exportierten Abbildungen,
die in der schriftlichen Ausarbeitung verwendet werden.

Die Grafiken wurden mit den Skripten im Ordner `Codes/` erzeugt
und dienen der didaktischen Erklärung zentraler Konzepte
der Bildverarbeitung und Schwingungsanalyse.

---

## 00_overview_all.png

Übersichtsdarstellung des gesamten MTF-Prinzips:

- Slanted-Edge-Target
- ESF (Edge Spread Function)
- LSF (Ableitung der ESF)
- MTF (FFT der LSF)

Zweck:
Gesamtzusammenhang der MTF-Bestimmung auf einer Abbildung.

---

## 01_slanted_edge_target.png

Synthetisches Slanted-Edge-Testbild.

Zweck:
Darstellung des schrägen Kantenprinzips
zur hochauflösenden Bestimmung der ESF.

---

## 02_ESF.png

Edge Spread Function (ESF).

Darstellung des Intensitätsverlaufs senkrecht zur Kante.

Zweck:
Veranschaulichung der optischen Unschärfe
als Übergangsbereich zwischen Schwarz und Weiß.

---

## 03_LSF.png

Line Spread Function (LSF).

Ergebnis der Ableitung der ESF.

Zweck:
Darstellung der Impulsantwort des Systems
im Ortsraum.

---

## 04_MTF.png

Modulation Transfer Function (MTF),
berechnet aus der Fourier-Transformation der LSF.

Zweck:
Frequenzabhängige Beschreibung
der Kontrastübertragung des Systems.

---

## mtf_grundidee_mtf_kurve.png

Schematische MTF-Kurve mit markierter MTF50.

Zweck:
Didaktische Erklärung der Kennzahl MTF50
als Maß für die Bildschärfe.

---

## laplace_beispiel_mehr_struktur.png

Beispielbild zur Veranschaulichung
der Laplace-Antwort.

Darstellung:
- Eingabebild mit mehreren geometrischen Strukturen
- Betrag der Laplace-Antwort
- Vorzeichenbehaftete Laplace-Antwort

Zweck:
Erklärung der Wirkungsweise des Laplace-Operators
auf Kanten und Strukturen.

---

## tenengrad_beispiel_gleiches_eingabebild.png

Demonstration der Tenengrad-Metrik
auf Basis desselben synthetischen Testbildes.

Darstellung:
- Sobel |Gx|
- Sobel |Gy|
- Gradientenenergie Gx² + Gy²

Zweck:
Veranschaulichung gradientenbasierter Schärfemetriken.

---

## 01_motion_blur_exposure_simple.png

Schematische Darstellung
der Bewegungsunschärfe während der Belichtung.

Inhalt:
- Zeitliche Integration
- Verschiebung während der Belichtungszeit
- Gemittelte Kante

Zweck:
Erklärung, wie Bewegungsunschärfe physikalisch entsteht.

---

## 02_sampling_aliasing_simple.png

Darstellung der zeitlichen Abtastung
und des Alias-Effekts.

Inhalt:
- Kontinuierliche Bewegung
- Diskrete Abtastung (Frames)
- Nyquist-Grenze
- Aliasing-Beispiel

Zweck:
Visualisierung der Abtasttheorie
im Kontext kamerabasierter Messungen.

---
