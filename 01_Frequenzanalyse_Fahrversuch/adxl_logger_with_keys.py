#!/usr/bin/env python3
"""
adxl_logger_with_keys.py

Single-owner logger for Arduino (ADXL345 sketch).
- Opens the real COM port (Arduino Serial) exclusively.
- Logs CSV lines to a timestamped file under ./logs/
- Echoes lines to console so you can watch live
- Lets you press keys to control the Arduino: s (start), x (pause), r (reset), c (calibrate)

Windows-only keystroke handling via msvcrt; on Linux/macOS it falls back to stdin (press Enter after a key).

Dependencies:
    pip install pyserial

Run:
    python adxl_logger_with_keys.py
"""

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import serial
import serial.tools.list_ports

# ===== User settings =====
BAUD        = 115200
OUTDIR      = "logs"
FLUSH_EVERY = 50
RETRY_SECONDS = 3
ECHO = True

FORCE_HEADER = None  


def pick_port():
    """Pick a likely Arduino/USB-Serial port automatically."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    # prefer Arduino-like / USB-serial chips
    keywords = ("arduino", "wch", "ch340", "cp210", "usb-serial", "ftdi", "silabs")
    for p in ports:
        desc = f"{p.device} {p.description} {p.manufacturer or ''}".lower()
        if any(k in desc for k in keywords):
            return p.device
    # else take the first
    return ports[0].device


def make_outfile() -> Path:
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(OUTDIR) / f"adxl_{ts}.csv"


def is_probable_header(s: str) -> bool:
    if "," not in s:
        return False
    cols = [c.strip() for c in s.strip().split(",")]
    if len(cols) < 3:
        return False
    # if first token is numeric, likely data not header
    try:
        float(cols[0])
        return False
    except ValueError:
        return True


def open_serial_auto(baud: int):
    while True:
        port = pick_port()
        if not port:
            print("[LOGGER] Kein serielles Gerät gefunden. USB ab/anklemmen? Warte…")
            time.sleep(RETRY_SECONDS)
            continue
        try:
            ser = serial.Serial(port, baud, timeout=1)
            print(f"[LOGGER] Verbunden: {port} @ {baud}")
            # small delay after opening (some boards auto-reset)
            time.sleep(2.0)
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            return ser
        except Exception as e:
            print(f"[LOGGER] Port {port} belegt/nicht erreichbar: {e}. Neuer Versuch in {RETRY_SECONDS}s…")
            time.sleep(RETRY_SECONDS)


def get_key_nonblocking():
    """Return a single key char if available, else None.
    On Windows uses msvcrt; on POSIX uses stdin (needs Enter).
    """
    try:
        import msvcrt
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            try:
                return ch.decode("utf-8").lower()
            except Exception:
                return None
        return None
    except ImportError:
        # POSIX fallback: non-blocking stdin with select
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip().lower()
            return line[:1] if line else None
        return None


def main():
    out_path = make_outfile()
    print(f"[LOGGER] Schreibe nach {out_path.resolve()}")
    print("[LOGGER] Steuerungstasten: s=start, x=pause, r=reset, c=calib, q=quit\n")

    # open out file
    f = out_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    header_written = False
    rows_since_flush = 0

    if FORCE_HEADER:
        writer.writerow([c.strip() for c in FORCE_HEADER.split(",")])
        f.flush()
        header_written = True

    ser = open_serial_auto(BAUD)

    try:
        while True:
            # read serial
            try:
                raw = ser.readline()
            except serial.SerialException as e:
                print(f"[LOGGER] Serial error: {e} (reconnect)")
                try: ser.close()
                except: pass
                time.sleep(RETRY_SECONDS)
                ser = open_serial_auto(BAUD)
                continue

            if raw:
                line = raw.decode("utf-8", errors="ignore").strip()
                if line:
                    if ECHO:
                        print(line)
                    if not header_written:
                        if FORCE_HEADER:
                            pass
                        elif is_probable_header(line):
                            writer.writerow([c.strip() for c in line.split(",")])
                            f.flush()
                            header_written = True
                        
                            continue
                        else:
                            parts = [p.strip() for p in line.split(",")]
                            if len(parts) >= 2:
                                writer.writerow([f"col{i+1}" for i in range(len(parts))])
                                f.flush()
                                header_written = True
                            else:
                                
                                pass
                    # write data 
                    parts = [p.strip() for p in line.split(",")]
                    writer.writerow(parts)
                    rows_since_flush += 1
                    if rows_since_flush >= FLUSH_EVERY:
                        f.flush()
                        rows_since_flush = 0

            # handle user key
            key = get_key_nonblocking()
            if key:
                if key in ("s","x","r","c"):
                    try:
                        ser.write(key.encode("utf-8"))
                        ser.flush()
                        print(f"[SEND] {key}")
                    except Exception as e:
                        print(f"[LOGGER] Sendefehler: {e}")
                elif key == "q":
                    print("[LOGGER] Quit (q).")
                    break
                else:
                 
                    pass

            # tiny sleep to reduce CPU
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[LOGGER] Stop (Ctrl+C).")

    finally:
        try:
            f.flush(); f.close()
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass
        print(f"[LOGGER] Gespeichert: {out_path.resolve()}")


if __name__ == "__main__":
    main()
