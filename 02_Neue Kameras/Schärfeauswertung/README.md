# Schärfeauswertung – Neue Kameras

Dieser Ordner enthält sämtliche Skripte zur schärfemetrischen Auswertung der Laboraufnahmen mit neuwertigen Kamerasystemen.

Ziel dieses Moduls ist die quantitative Analyse frequenzabhängiger Schärfeveränderungen unter definierten Vibrationsanregungen.

---

## Inhalt und Funktion der Skripte

### 1. Laplace_Cams.py

Berechnet die Laplace-Varianz für jede einzelne Kamera separat.

Funktion:
- Frameweise Berechnung der Laplace-Varianz
- ROI-basierte Auswertung (zentraler Bildausschnitt)
- Optionale Median- oder Mittelwertbildung
- Speicherung von:
  - Summary-CSV pro Kamera
  - Frequenzdiagrammen
  - Normierten Verläufen
  - Mean-vs-Median-Diagnoseplots
  - Verteilungsdiagrammen (Violinplots)

Zweck:
Analyse des frequenzabhängigen Schärfeverhaltens pro Kamera.

---

### 2. Laplace_gesamt.py

Aggregiert die Laplace-Ergebnisse über mehrere Kameras.

Funktion:
- Einlesen der einzelnen Summary-CSVs
- Entfernen fehlerhafter Kamera
- Normierung auf Referenzfrequenz (z. B. 2 Hz)
- Medianbildung über Kameras
- Erstellung aggregierter relativer Schärfeverläufe

Zweck:
Darstellung eines robusten Gesamtschärfetrends unabhängig von einzelnen Kameravarianzen.

---

### 3. Tenengrad_Cams.py

Berechnet die Tenengrad-Metrik pro Kamera.

Funktion:
- Sobel-basierte Gradientenberechnung
- Gradientenergie (Tenengrad)
- ROI-Auswertung
- Median/Mittelwert-Bildung
- Speicherung von:
  - Summary-CSV
  - Frequenzdiagrammen
  - Normierten Verläufen
  - Mean-vs-Median-Diagnose
  - Verteilungsplots

Zweck:
Gradientenbasierte Bewertung vibrationsbedingter Schärfeverluste.

---

### 4. Tenengrad_gesamt.py

Aggregiert Tenengrad-Ergebnisse über mehrere Kameras.

Funktion:
- Normierung auf Referenzfrequenz
- Medianbildung über Kameras
- Erstellung aggregierter relativer Schärfeverläufe
- Export kombinierter CSV-Dateien

Zweck:
Vergleich mit Laplace-Trends und Bewertung der Robustheit der Schärfemetriken.

---

### 5. normal_aggregiert.py

Erstellt kombinierte aggregierte Diagramme für Laplace und Tenengrad.

Funktion:
- Einlesen aggregierter CSVs
- Normierung auf Referenzfrequenz
- Medianbildung über Kameras
- Vergleich Sinus vs. Rechteck
- Export finaler Diagramme für Arbeit/Kolloquium

Zweck:
Direkter Vergleich beider Schärfemetriken im Frequenzverlauf.

---

### 6. umgekehrt.py

Auswertung der Umkehrmessung (bewegtes Target, stationäre Kamera).

Funktion:
- ROI-basiertes Cropping (fest definierte Region)
- Berechnung von Laplace und Tenengrad
- Vergleich mit ursprünglichen Messreihen
- Normierte Frequenzdarstellung
- Analyse von Ruhe-Messungen
- Diagnoseplots

Zweck:
- Trennung optischer Effekte von rein bewegungsbedingten Einflüssen
- Validierung der physikalischen Konsistenz der Schärfemessung

---
