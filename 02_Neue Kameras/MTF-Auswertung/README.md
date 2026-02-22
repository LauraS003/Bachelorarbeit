# MTF-Auswertung – Neue Kameras

Dieser Ordner enthält Skripte zur Bestimmung und Auswertung der Modulation Transfer Function (MTF) für Laboraufnahmen mit neuwertigen Kamerasystemen.

Ziel ist die frequenzabhängige Charakterisierung der Detailübertragung unter definierten Vibrationsanregungen.

---

## Enthaltene Skripte

### `mtf_green.py`

Interaktives Hauptskript zur MTF-Berechnung aus Videodaten mittels Slanted-Edge-Verfahren.

Funktionen:

- Automatische Suche nach Frequenzordnern (z. B. 2Hz, 5Hz, …, Ruhe)
- Auswahl eines Referenzwinkels anhand des Ruhe-Videos
- Interaktive ROI-Positionierung und Kantenausrichtung
- ESF- → LSF- → MTF-Berechnung
- Automatische Bestimmung von:
  - MTF50 
  - MTF-Werten bei festen Ortsfrequenzen (z. B. 0.05, 0.10, 0.20 cy/px)
- Frameweise Auswertung und statistische Kennwerte:
  - Mittelwert
  - Median
  - Standardabweichung
- Export:
  - MTF-Kurvenplots
  - Rotierte ROI-Bilder
  - Summary-CSV-Datei

Das Skript dient als zentrale MTF-Auswerteroutine für alle Frequenzmessungen.

---

### `mtf_final_diagramme.py`

Erstellt aggregierte Diagramme auf Basis der erzeugten Summary-CSV-Datei.

Dargestellte Kenngrößen:

- MTF50 vs. Anregungsfrequenz
- MTF@0.05 vs. Anregungsfrequenz
- Standardabweichung der MTF-Werte (zeitliche Instabilität)

Die Diagramme werden als PNG-Dateien gespeichert und in der schriftlichen Arbeit verwendet.

---

