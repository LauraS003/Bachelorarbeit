# Bachelorarbeit – Einfluss von Vibrationen auf die Bildqualität hochauflösender Fahrzeugkameras

Autorin: Laura Seher  
Studiengang: Technische Informatik  
Thema: Einfluss von Vibrationen auf die Bildqualität hochauflösender Kameras

---

## Repository-Struktur

Dieses Repository enthält sämtliche Auswerte-Skripte, Validierungen und Abbildungen, die im Rahmen der Bachelorarbeit verwendet wurden.

01_Frequenzanalyse_Fahrversuch/
02_Auswertung_Laborversuche/
03_Abbildungen/

---

# 01_Frequenzanalyse_Fahrversuch

Enthält die Mess- und Auswerte-Skripte zur realen Frequenzcharakterisierung während einer Fahrzeugfahrt.

## Inhalt

- `adxl_logger_with_keys.py`  
  Aufzeichnung der Beschleunigungsdaten (ADXL-Sensor)

- `fahrt_auswertung.py`  
  Frequenzanalyse der aufgezeichneten Beschleunigungszeitreihen  
  → FFT  
  → Leistungsdichtespektrum (PSD)  
  → Identifikation relevanter Anregungsbereiche

## Ziel

Bestimmung realer fahrzeuginduzierter Frequenzbereiche als Referenz für die Laboruntersuchungen.

---

# 02_Auswertung_Laborversuche

Enthält sämtliche Auswerte-Skripte der Laboruntersuchungen.

## Struktur

01_Validierung/

02_Neue Kameras/

03_Gealterte Kameras/

04_Schlechtwegfahrt/


## 1. Validierung

Überprüfung des Versuchsaufbaus:

- Einfluss der Zusatzlinse
- Höhenprüfung (Unten / Mitte / Oben)
- Sicherstellung reproduzierbarer Schärfemessung

## 2. Neue Kameras

Auswertung der Laborversuche mit neuwertigen Kamerasystemen:

- Laplace-Varianz
- Tenengrad
- Optical Flow
- MTF-Auswertung (Slanted-Edge-Verfahren)
- Umkehrmessung (Target bewegt, Kamera fixiert)
- Aggregierte Frequenzdarstellung

Ziel: Referenzcharakterisierung des frequenzabhängigen Schärfeverhaltens.

## 3. Gealterte Kameras

Vergleich gealterter Kamerasysteme mit Neuzustand:

- Einzelanalysen
- Laplace- und Tenengrad-Auswertung
- Direktvergleich alt vs. neu

Ziel: Untersuchung möglicher altersbedingter Schärfeverluste.

## 4. Schlechtwegfahrt

Auswertung einer real aufgezeichneten Schlechtwegfahrt:

- Frameweise Schärfeanalyse
- Untersuchung von Kurven- und Stoßereignissen
- Identifikation erhöhter Unschärfe bei lateralen Anregungen

---

# 03_Abbildungen

Enthält alle für die schriftliche Arbeit und das Kolloquium erzeugten Darstellungen.

## Codes/

Python-Skripte zur Erzeugung didaktischer und methodischer Abbildungen:

- Slanted-Edge-Visualisierung
- ESF / LSF / MTF-Darstellungen
- Tenengrad- und Laplace-Erklärbilder
- Abtastung & Aliasing
- Bewegungsunschärfe während Belichtung

## Darstellungen/

Exportierte PNG-Abbildungen für schriftliche Ausarbeitung

---

# Technische Grundlage

Programmiersprache: Python  
Bibliotheken:  
- OpenCV  
- NumPy  
- Matplotlib  
- SciPy  

---
