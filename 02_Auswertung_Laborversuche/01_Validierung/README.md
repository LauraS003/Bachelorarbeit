# 1. Validierung – Einfluss der Zusatzlinse (Höhenprüfung)

## Ziel

In diesem Versuch wurde überprüft, ob die zusätzlich eingesetzte Linse bzw. die mechanische Anordnung der Kamera einen lageabhängigen Einfluss auf die gemessene Bildschärfe hat.

Hierzu wurde die Kamera in drei vertikalen Positionen montiert:

- **Unten**
- **Mitte**
- **Oben**

Ziel war es sicherzustellen, dass keine systematische Unschärfe durch die zusätzliche Linse oder durch geometrische Effekte entsteht.

---

## Datei: `check_lens_effect.py`

### Funktion

Das Skript berechnet für alle Bilder der drei Höhenpositionen die **Varianz des Laplace-Operators** als objektive Schärfemetrik.

### Ablauf

1. Einlesen aller Bilder (.png)
2. Optionales ROI-Cropping 
3. Berechnung der Schärfe:
- Laplace-Filter
- Varianz der Laplace-Antwort
4. Pro Position:
- Mittelwert
- Standardabweichung
- Minimum / Maximum
- Normierung auf Referenz „Mitte“
- Prozentuale Abweichung
5. Ausgabe:
- `schaerfe_ergebnisse.csv`
- `schaerfe_ergebnisse_stat_norm.csv`
- Balkendiagramm (`sharpness_plot.png`)
- optionales Fehlerlog

---

## Interpretation

Wenn die normierten Werte der Positionen „Unten“ und „Oben“ nur geringe Abweichungen zur Referenz „Mitte“ zeigen, kann davon ausgegangen werden, dass:

- keine lageabhängige Zusatzunschärfe durch die Linse entsteht
- der Laboraufbau geometrisch stabil ist
- die Zusatzoptik keinen relevanten Einfluss auf die Schärfemessung hat

Signifikante systematische Abweichungen würden hingegen auf optische oder mechanische Effekte hindeuten.

---
