# Codes zur Erstellung von Abbildungen

Dieser Ordner enthält Python-Skripte zur Generierung der in der schriftlichen Arbeit verwendeten schematischen Darstellungen.

Die Skripte dienen ausschließlich der Visualisierung theoretischer Zusammenhänge
(z. B. Laplace-Operator, Tenengrad, MTF, Abtasttheorem) und sind nicht Bestandteil
der experimentellen Messauswertung.

---

## kante_mtf.py

Erzeugt eine schematische Darstellung des Slanted-Edge-Verfahrens
zur Bestimmung der Modulation Transfer Function (MTF).

Funktion:
- Generierung eines synthetischen Kantenbildes
- Simulation einer Unschärfe
- Darstellung von ESF (Edge Spread Function)
- Ableitung zur LSF (Line Spread Function)
- Fourier-Transformation zur MTF

Zweck:
Didaktische Visualisierung des Zusammenhangs
Kante → ESF → LSF → MTF.

---

## laplace_antwort.py

Erzeugt ein synthetisches Testbild mit mehreren geometrischen Strukturen und berechnet die Laplace-Antwort.

Funktion:
- Eigenimplementierte 2D-Faltung
- Anwendung eines diskreten 3×3-Laplace-Kernels
- Darstellung von:
  - Eingabebild
  - Betrag der Laplace-Antwort (Kantenstärke)
  - Vorzeichenbehaftete Laplace-Antwort

Zweck:
Anschauliche Erklärung der Wirkungsweise des Laplace-Operators
und dessen Reaktion auf Kanten- und Strukturänderungen.

---

## mtf_bilder.py

Erzeugt schematische Darstellungen zur Grundidee der MTF
über sinusförmige Ortsfrequenzmuster.

Funktion:
- Generierung von Sinusmustern unterschiedlicher Ortsfrequenz
- Simulation eines optischen Systems durch Gauß-Blur
- Berechnung der Kontrastübertragung (RMS-Kontrast)
- Darstellung:
  - Original vs. verschwommen
  - Kontrastübertragung über Ortsfrequenz
  - Schematische MTF-Kurve mit markierter MTF50

Zweck:
Didaktische Verdeutlichung der frequenzabhängigen Kontrastübertragung
eines optischen Systems.

---

## tenengrad_antwort.py

Demonstriert die Funktionsweise der Tenengrad-Metrik
auf Basis eines synthetischen Testbildes mit mehreren geometrischen Strukturen.

Funktion:
- Berechnung der Sobel-Gradienten Gx und Gy
- Bildung der Gradientenenergie (Gx² + Gy²)
- Visualisierung von:
  - Eingabebild
  - |Gx|
  - |Gy|
  - Gradientenenergie (Tenengrad)

Zweck:
Veranschaulichung, wie gradientenbasierte Schärfemetriken
auf lokale Intensitätsänderungen reagieren.

---

## abtastung.py

Erzeugt eine grafische Darstellung des Nyquist-Shannon-Abtasttheorems
für periodische Kamerabewegungen.

Funktion:
- Simulation einer kontinuierlichen harmonischen Bewegung
- Diskrete Abtastung mit definierter Bildrate (fps)
- Darstellung von:
  - Unter-Nyquist-Fall (korrekte Abtastung)
  - Über-Nyquist-Fall (Aliasing)

Zweck:
Visualisierung der zeitlichen Abtastung einer Kamerabewegung
und Erklärung des Alias-Effekts bei zu geringer Bildrate.

---
