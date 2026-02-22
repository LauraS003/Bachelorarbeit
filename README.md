# 02_Auswertung_Laborversuche

Dieser Ordner enthält sämtliche Auswerteskripte zur Analyse der im Labor aufgenommenen Videoaufnahmen unter definierten Vibrationsanregungen.

Die Laborversuche dienen der systematischen Untersuchung des Einflusses mechanischer Schwingungen auf die Bildschärfe von Fahrzeugaußenkameras. Im Gegensatz zum realen Fahrversuch werden hier kontrollierte Anregungsformen (z. B. Sinus- oder Rechteckschwingungen) mit definierten Frequenzen eingesetzt.

Die Auswertung erfolgt mithilfe verschiedener bildanalytischer Verfahren:

- Varianz des Laplace-Operators
- Tenengrad-Metrik
- Modulation Transfer Function (MTF)
- Optischer Fluss

---

## Ordnerstruktur

### 1. Validierung

Enthält Skripte zur Validierung des Laboraufbaus hinsichtlich möglicher optischer Einflussfaktoren.

Ziel:
- Überprüfung, ob die Positionierung der Kamera bzw. Linse selbst systematische Unschärfe erzeugt
- Ausschluss aufbaubedingter Artefakte
- Sicherstellung, dass gemessene Schärfeverluste primär auf Vibrationsanregungen zurückzuführen sind

Im Rahmen dieser Validierung wurde untersucht, ob bestimmte Linsen- oder Kamerapositionen unabhängig von einer mechanischen Anregung zu veränderten Schärfewerten führen. Dadurch wird sichergestellt, dass der Versuchsaufbau keine positionsabhängigen Unschärfeeffekte einbringt, die die Interpretation der Vibrationsmessungen verfälschen könnten.

---

### 2. Neue Kameras

### Neue Kameras

Beinhaltet die Auswertung von Laboraufnahmen mit neuwertigen Kamerasystemen unter definierten Vibrationsanregungen.

Ziel:
- Referenzcharakterisierung des Schärfeverhaltens neuwertiger Kameras
- Untersuchung frequenzabhängiger Schärfeverluste
- Vergleich unterschiedlicher Anregungsformen (z. B. Sinus, Rechteck)
- Analyse des Zusammenhangs zwischen mechanischer Bewegung und bildseitiger Verschiebung

Neben klassischen No-Reference-Schärfemetriken (Laplace-Varianz, Tenengrad) werden in diesem Bereich zusätzlich folgende Verfahren eingesetzt:

- **Modulation Transfer Function (MTF)** zur frequenzselektiven Bewertung der Detailübertragung
- **Optischer Fluss** zur Quantifizierung der bildseitigen Relativbewegung zwischen aufeinanderfolgenden Frames
- **Umkehrmessung** zur Überprüfung der Richtungsabhängigkeit der Schärfemetriken und zur Absicherung symmetrischer Anregungseffekte

Dieser Ordner bildet damit die zentrale Referenzanalyse für den Vergleich mit gealterten Kamerasystemen.

---

### 3. Gealterte Kameras

Enthält Auswertungen von Kameras mit bereits gealterten oder optisch degradierten Eigenschaften.

Ziel:
- Untersuchung des Einflusses optischer Alterung auf vibrationsbedingte Schärfeverluste
- Vergleich mit neuwertigen Kameras
- Bewertung möglicher Interaktionseffekte zwischen optischer Systemqualität und Bewegung

---

### 4. Schlechtwegfahrt

Enthält Skripte zur Analyse einer real aufgenommenen Schlechtwegfahrt, deren Videodaten hinsichtlich vibrationsbedingter Bildunschärfe ausgewertet wurden.

Ziel:
- Quantifizierung des Schärfeverlaufs während einer realen Fahrbahnanregung
- Identifikation von Fahrzuständen mit erhöhter Unschärfe
- Untersuchung des Zusammenhangs zwischen Fahrzeugbewegung und gemessenen Schärfemetriken

Im Rahmen der Auswertung wurden insbesondere zeitliche Verläufe der Schärfemetriken analysiert, um festzustellen, in welchen Fahrsituationen erhöhte Bewegungsunschärfe auftritt. Dabei zeigte sich, dass insbesondere Kurvenfahrten zu deutlich ausgeprägteren Schärfeverlusten führen können als geradlinige Fahrtabschnitte.

---