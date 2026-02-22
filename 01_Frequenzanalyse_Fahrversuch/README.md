# 01_Frequenzanalyse_Fahrversuch

Dieser Ordner enthält die Skripte zur Aufnahme und frequenzanalytischen Auswertung der Beschleunigungsdaten während realer Fahrversuche. Ziel ist die Charakterisierung fahrzeuginduzierter Schwingungsanregungen im relevanten Frequenzbereich.

Die Messungen dienen als experimentelle Grundlage für die Analyse vibrationsbedingter Bildunschärfe.

---

## 1. adxl_logger_with_keys.py

Dieses Skript dient zur **Aufzeichnung der Beschleunigungsdaten** eines ADXL345-Sensors (Arduino-basiert) während der Fahrt.

### Zweck
- Serielle Kommunikation mit dem Arduino
- Echtzeit-Logging der Sensordaten
- Speicherung als CSV-Datei mit Zeitstempel
- Steuerung des Arduino über Tastatureingaben

### Funktionsweise

- Automatische Erkennung des seriellen COM-Ports
- Aufbau einer seriellen Verbindung mit 115200 Baud
- Einlesen der vom Arduino gesendeten CSV-Daten
- Speicherung der Daten im Ordner `logs/`
- Echtzeit-Anzeige der Messwerte im Terminal
- Steuerung per Tastatur:
  - `s` → Start
  - `x` → Pause
  - `r` → Reset
  - `c` → Kalibrierung
  - `q` → Beenden

Die erzeugten CSV-Dateien enthalten typischerweise:
- Zeitstempel (`t_ms`)
- Beschleunigungskomponenten (`g_x`, `g_y`, `g_z`)

### Ziel der Aufnahme

Erfassung realer Beschleunigungszeitreihen zur späteren:
- Frequenzanalyse
- Berechnung der Power Spectral Density (PSD)

---

## 2. fahrt_auswertung.py

Dieses Skript dient der **frequenzanalytischen Auswertung** der während der Fahrt aufgenommenen ADXL-Daten.

### Zweck
- Einlesen der aufgezeichneten CSV-Datei
- Signalaufbereitung
- Berechnung des Leistungsdichtespektrums (PSD)
- Visualisierung im Zeit- und Frequenzbereich

### Verarbeitungsschritte

1. Einlesen der CSV-Datei mit `pandas`
2. Berechnung der resultierenden Beschleunigungsgröße:

   a_mag = sqrt(gx² + gy² + gz²)

3. Anwendung eines Hochpassfilters (2. Ordnung, 2 Hz)
   - Entfernung quasistatischer Anteile (z.B. Gravitation)
   - Fokus auf dynamische Schwingungsanteile

4. Berechnung der Power Spectral Density (PSD)
   - Methode: Welch-Verfahren
   - Abtastrate: 50 Hz

5. Darstellung:
   - Gefilterter Beschleunigungsverlauf im Zeitbereich
   - Leistungsdichtespektrum im Frequenzbereich

### Ziel der Auswertung

- Identifikation dominanter Anregungsfrequenzen
- Bestimmung relevanter Frequenzbereiche für die Bildanalyse
- Vergleich mit Literaturwerten typischer NVH-Phänomene
- Ableitung geeigneter Labor-Anregungsfrequenzen

---

## Wissenschaftlicher Kontext

Die hier gewonnenen Frequenzspektren bilden die experimentelle Grundlage zur:

- Einordnung fahrzeugtypischer Schwingungsbereiche
- Bewertung der Relevanz bestimmter Frequenzen für vibrationsbedingte Bewegungsunschärfe
- Validierung der in der Arbeit verwendeten Laboranregungen

---

## Abhängigkeiten

Erforderliche Python-Pakete:

- pandas
- numpy
- scipy
- matplotlib
- pyserial

Installation z. B. über:

pip install pandas numpy scipy matplotlib pyserial

---
