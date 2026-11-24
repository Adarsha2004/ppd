"""Wiring diagram: ESP32-S3-N16R8 DevKit  <->  7Semi MLX90640 thermal camera.

Run:  python3 make_wiring_diagram.py   ->  wiring_diagram.png + wiring_diagram.svg
Pin assignments match firmware/thermal_streamer/thermal_streamer.ino.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

HERE = os.path.dirname(os.path.abspath(__file__))

# ESP32-S3-DevKitC-1 header J1, listed from the USB end (USB connectors at the top,
# as on the breadboard).
J1 = ["G", "5V", "14", "13", "12", "11", "10", "9", "46", "3", "8", "18",
      "17", "16", "15", "7", "6", "5", "4", "RST", "3V3", "3V3"]
SENSOR = ["SDA", "SCL", "GND", "3V3", "VIN"]

# (esp pin, sensor pin, colour, label)
WIRES = [
    ("3V3", "VIN", "#d62728", "3.3 V power"),
    ("G",   "GND", "#222222", "Ground"),
    ("8",   "SDA", "#1f77b4", "I2C data  (SDA)"),
    ("9",   "SCL", "#e6b800", "I2C clock (SCL)"),
]

fig, ax = plt.subplots(figsize=(15, 10.2))
ax.set_xlim(0, 15)
ax.set_ylim(-0.6, 9.6)
ax.axis("off")

# ---------------------------------------------------------------- ESP32 board
bx, by, bw, bh = 1.2, 0.55, 2.6, 7.55
ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.12",
                            fc="#1b1b1b", ec="black", lw=1.5))
for i, lbl in enumerate(("USB", "COM")):                         # two USB-C ports
    ax.add_patch(Rectangle((bx + 0.45 + i * 1.05, by + bh - 0.05), 0.6, 0.38,
                           fc="#b0b0b0", ec="black", lw=1))
    ax.text(bx + 0.75 + i * 1.05, by + bh + 0.47, lbl, ha="center", fontsize=12.3,
            fontweight="bold" if lbl == "USB" else "normal")
ax.add_patch(Rectangle((bx + 0.15, by + 0.45), 1.15, 2.4, fc="#cfcfcf", ec="#888", lw=1))
ax.text(bx + 0.725, by + 1.65, "ESP32-S3\nWROOM-1\nN16R8", ha="center", va="center", fontsize=12.3)
ax.add_patch(Circle((bx + 0.7, by + 4.6), 0.16, fc="#7fd0ff", ec="white", lw=1))
ax.text(bx + 0.7, by + 4.2, "RGB LED\n(GPIO 48)", ha="center", va="top", color="white", fontsize=10.9)
ax.text(bx + bw / 2, by + bh + 0.95, "ESP32-S3-N16R8 DevKit", ha="center",
        fontsize=18.8, fontweight="bold")

pin_y = {}
top, step = by + bh - 0.42, (bh - 0.75) / (len(J1) - 1)
used = {w[0] for w in WIRES} | {"4"}
for i, name in enumerate(J1):
    y = top - i * step
    pin_y.setdefault(name, y)
    if name == "3V3" and i == len(J1) - 2:                       # wire goes to the first 3V3
        pin_y["3V3"] = y
    px = bx + bw - 0.2
    on = name in used and (name != "3V3" or i == len(J1) - 2)
    ax.add_patch(Circle((px, y), 0.075, fc="#f5c542" if on else "#9a9a9a", ec="black", lw=0.6))
    ax.text(px - 0.17, y, name if name in ("G", "5V", "RST", "3V3") else f"GPIO{name}"
            if name.isdigit() else name, ha="right", va="center", color="white",
            fontsize=10.9, fontweight="bold" if on else "normal")

# ---------------------------------------------------------------- MLX90640 breakout
sx, sy, sw, sh = 10.2, 3.2, 3.3, 3.4
ax.add_patch(FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=0.02,rounding_size=0.1",
                            fc="#14307a", ec="black", lw=1.5))
for cx, cy in ((sx + 0.25, sy + 0.25), (sx + sw - 0.25, sy + 0.25),
               (sx + 0.25, sy + sh - 0.25), (sx + sw - 0.25, sy + sh - 0.25)):
    ax.add_patch(Circle((cx, cy), 0.14, fc="white", ec="#888", lw=1))
ax.add_patch(Circle((sx + 1.85, sy + sh / 2), 0.5, fc="black", ec="#c9a227", lw=3))
ax.text(sx + 1.85, sy + sh / 2 - 0.75, "MLX90640\n110° x 75°", ha="center", va="top",
        color="white", fontsize=11.6)
ax.text(sx + sw / 2, sy + sh + 0.55, "7Semi MLX90640 Thermal Camera", ha="center",
        fontsize=18.8, fontweight="bold")
ax.text(sx + sw / 2, sy + sh + 0.2, "32 x 24 IR array  |  I2C address 0x33  |  3.3 V",
        ha="center", fontsize=13.0, color="#444")

sens_y = {}
for i, name in enumerate(SENSOR):
    y = sy + sh - 0.75 - i * 0.48
    sens_y[name] = y
    px = sx + 0.2
    ax.add_patch(Circle((px, y), 0.075, fc="#f5c542" if name != "3V3" else "#9a9a9a",
                        ec="black", lw=0.6))
    ax.text(px + 0.17, y, name, ha="left", va="center", color="white", fontsize=12.3,
            fontweight="bold")

# ---------------------------------------------------------------- wires (orthogonal)
x_start = bx + bw - 0.2
x_end = sx + 0.2
lanes = {"VIN": 7.5, "GND": 7.0, "SDA": 6.0, "SCL": 6.5}       # vertical run per wire
for esp, sen, col, label in WIRES:
    y0, y1, xl = pin_y[esp], sens_y[sen], lanes[sen]
    ax.plot([x_start, xl, xl, x_end], [y0, y0, y1, y1], color=col, lw=3.2,
            solid_capstyle="round", solid_joinstyle="round", zorder=3)
    below = esp == "3V3"
    ax.text((x_start + xl) / 2 + 0.25, y0 - 0.1 if below else y0 + 0.09, f"{esp if not esp.isdigit() else 'GPIO' + esp}"
            f" -> {sen}", color=col if col != "#e6b800" else "#a88600", fontsize=12.3,
            fontweight="bold", ha="center", va="top" if below else "bottom", zorder=4)

# optional alert output
ya = pin_y["4"]
ax.plot([x_start, 5.0], [ya, ya], color="#2ca02c", lw=2.2, ls=(0, (4, 2)), zorder=3)
ax.add_patch(Rectangle((5.0, ya - 0.11), 0.55, 0.22, fc="#e8d8b0", ec="black", lw=0.8, zorder=4))
ax.text(5.275, ya - 0.2, "220 Ω", ha="center", va="top", fontsize=10.9)
ax.plot([5.55, 6.0], [ya, ya], color="#2ca02c", lw=2.2, ls=(0, (4, 2)), zorder=3)
ax.add_patch(Circle((6.15, ya), 0.15, fc="#ff5555", ec="black", lw=0.8, zorder=4))
ax.plot([6.3, 6.55], [ya, ya], color="#2ca02c", lw=2.2, ls=(0, (4, 2)), zorder=3)
ax.text(6.6, ya, "GND", va="center", fontsize=11.6, color="#2ca02c")
ax.text(4.0, ya + 0.2, "optional alert LED / buzzer (HIGH = change)",
        va="bottom", fontsize=11.6, color="#2ca02c")

# USB to laptop
ax.text(bx + bw + 0.45, by + bh + 0.55, "USB-C cable to MacBook (port labelled USB)\n"
        "frames out @ ~8 fps  |  'D'/'C' detection result back",
        fontsize=12.3, va="center", ha="left", color="#2b6ea6")

# ---------------------------------------------------------------- connection table
rows = [("ESP32-S3 pin", "MLX90640 pin", "Wire", "Function")] + \
       [(e if not e.isdigit() else f"GPIO {e}", s, c, l) for e, s, c, l in WIRES]
tx, ty = 7.9, 2.45
for r, (a, b, c, d) in enumerate(rows):
    y = ty - r * 0.45
    head = r == 0
    ax.text(tx, y, a, fontsize=13.0, fontweight="bold" if head else "normal", va="center")
    ax.text(tx + 1.85, y, b, fontsize=13.0, fontweight="bold" if head else "normal", va="center")
    if head:
        ax.text(tx + 3.6, y, c, fontsize=13.0, fontweight="bold", va="center")
    else:
        ax.plot([tx + 3.6, tx + 4.2], [y, y], color=c, lw=5)
    ax.text(tx + 4.5, y, d, fontsize=13.0, fontweight="bold" if head else "normal", va="center")
ax.plot([tx - 0.1, tx + 6.9], [ty - 0.2, ty - 0.2], color="#999", lw=0.8)

ax.text(7.5, -0.5,
        "Power the sensor from 3V3 only (not 5V).   Sensor pin '3V3' stays unconnected (VIN feeds its regulator).\n"
        "I2C runs at 400 kHz; the firmware auto-detects SDA/SCL if they are swapped.\n"
        "Onboard LED: blue = waiting for laptop, green = clear, red = change, purple blink = sensor not found.",
        ha="center", va="bottom", fontsize=12, color="#333", linespacing=1.5)

fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(os.path.join(HERE, f"wiring_diagram.{ext}"), dpi=200, bbox_inches="tight",
                facecolor="white")
print("saved wiring_diagram.png / .svg ->", HERE)
