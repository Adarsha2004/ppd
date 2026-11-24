"""Real-time MSCP-Mamba thermal change detection.

  ESP32-S3 + MLX90640  --(USB serial / WiFi)-->  this script on the Mac (M-series GPU)
                       <--- 'D' / 'C' ---------  (drives the board's RGB LED / alert pin)

Examples
  python run_realtime.py --weights weights/best__cnn_mamba_balanced.pt                 # USB, auto port
  python run_realtime.py --weights ... --source tcp:thermal-cam.local                  # WiFi
  python run_realtime.py --weights ... --source sim                                    # no hardware
  python run_realtime.py --weights ... --source replay:recordings/rec_XXXX.npz

Keys (video window):  q quit | b capture background (empty scene) | x clear background
                      r start/stop recording | s snapshot | f flip | n normalisation
                      [ / ] threshold -/+ | 1 2 4 8 6 sensor refresh (Hz per subpage)
"""
import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import torch

import model as mscp
from stream import open_source

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Sensor -> model input
# --------------------------------------------------------------------------- #
class Preprocessor:
    """32x24 degC -> [1,3,240,320] float in [0,1], like a CDnet thermal frame
    (8-bit, hot = bright, auto-ranged, grey replicated to RGB).

    norm='auto' : robust per-frame range (2nd..98th percentile), EMA-smoothed so the
                  brightness doesn't pump, with a minimum span so an empty room's
                  noise isn't stretched to full contrast.
    norm='fixed': linear map of [tmin, tmax] degC -> [0, 255]."""

    def __init__(self, size=(240, 320), norm="auto", tmin=18.0, tmax=36.0,
                 min_span=8.0, ema=0.15, flip="none", spatial=True, temporal=0.5):
        self.H, self.W = size
        self.norm, self.tmin, self.tmax = norm, tmin, tmax
        self.min_span, self.ema, self.flip = min_span, ema, flip
        self.spatial = spatial      # 3x3 binomial [1 2 1]: exactly cancels the chess-mode checkerboard
        self.temporal = temporal    # 0 = off; 0.5 = average new frame 50/50 with the previous output
        self._lo = self._hi = None
        self._prev = None
        self.bg = None              # fixed-pattern offset map (degC), see start_background()
        self.rel_span = 4.0         # with a background: degC above room level that maps to white
        self.delta = None           # (H,W) degC above room level, available once a background exists
        self._bg_acc, self._bg_n = [], 0

    def start_background(self, n=16):
        """Average the next n frames (scene must be EMPTY) and from then on subtract their
        fixed pattern: lens-edge warming + static warm objects. Only changes remain."""
        self._bg_acc, self._bg_n = [], n

    def clear_background(self):
        self.bg, self._bg_acc, self._bg_n = None, [], 0

    @property
    def calibrating(self):
        return self._bg_n > 0

    def denoise(self, t):
        if self.spatial:
            t = cv2.sepFilter2D(t.astype(np.float32), -1, np.array([.25, .5, .25], np.float32),
                                np.array([.25, .5, .25], np.float32), borderType=cv2.BORDER_REPLICATE)
        if self.temporal > 0:
            if self._prev is not None and self._prev.shape == t.shape:
                t = self.temporal * self._prev + (1 - self.temporal) * t
            self._prev = t
        return t

    def range(self, temps):
        if self.norm == "fixed":
            return self.tmin, self.tmax
        if self.bg is not None and self.rel_span > 0:
            # background known: room level -> dark, +rel_span degC -> white (CDnet-like:
            # bright people on a dark scene), regardless of how warm the room is
            med = float(np.median(temps))
            lo, hi = med - 1.0, med + self.rel_span
            if self._lo is None:
                self._lo, self._hi = lo, hi
            else:
                self._lo += self.ema * (lo - self._lo)
                self._hi += self.ema * (hi - self._hi)
            return self._lo, self._hi
        lo, hi = np.percentile(temps, [2, 98])
        if hi - lo < self.min_span:
            mid = (hi + lo) / 2
            lo, hi = mid - self.min_span / 2, mid + self.min_span / 2
        if self._lo is None:
            self._lo, self._hi = lo, hi
        else:
            self._lo += self.ema * (lo - self._lo)
            self._hi += self.ema * (hi - self._hi)
        return self._lo, self._hi

    def __call__(self, temps):
        t = temps
        if self.flip in ("h", "hv"):
            t = t[:, ::-1]
        if self.flip in ("v", "hv"):
            t = t[::-1, :]
        t = np.ascontiguousarray(t, dtype=np.float32)
        if self._bg_n:
            self._bg_acc.append(t.copy())
            if len(self._bg_acc) >= self._bg_n:
                bg = np.mean(self._bg_acc, axis=0)
                self.bg = (bg - np.median(bg)).astype(np.float32)   # keep absolute level
                self._bg_acc, self._bg_n = [], 0
                self._lo = self._hi = None                          # re-learn display range
                print(f"background captured (pattern range {self.bg.min():+.1f}..{self.bg.max():+.1f} C)")
        if self.bg is not None and self.bg.shape == t.shape:
            t = t - self.bg
        t = self.denoise(t)
        if self.bg is not None:
            d = (t - np.median(t)).astype(np.float32)
            self.delta = cv2.resize(d, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        else:
            self.delta = None
        lo, hi = self.range(t)
        g = np.clip((t - lo) / max(hi - lo, 1e-3), 0, 1).astype(np.float32)
        g = cv2.resize(g, (self.W, self.H), interpolation=cv2.INTER_CUBIC)
        g = np.clip(g, 0, 1)
        gray8 = (g * 255).astype(np.uint8)               # what the model "sees"
        x = torch.from_numpy(gray8.astype(np.float32) / 255.0)[None, None].repeat(1, 3, 1, 1)
        return x, gray8, np.ascontiguousarray(t), (lo, hi)


# --------------------------------------------------------------------------- #
# Decision logic
# --------------------------------------------------------------------------- #
def clean_mask(mask, speck):
    """Remove blobs smaller than `speck` (fraction of frame). Returns the cleaned
    mask, its foreground fraction, and the largest blob's fraction."""
    m = mask.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m.astype(bool), 0.0, 0.0
    areas = st[1:, cv2.CC_STAT_AREA] / m.size
    keep = np.concatenate([[False], areas >= speck])
    out = keep[lab]
    return out, float(out.mean()), float(areas.max())


# --------------------------------------------------------------------------- #
# Debounce
# --------------------------------------------------------------------------- #
class Detector:
    def __init__(self, min_area=0.03, on_frames=2, off_frames=4):
        self.min_area, self.on_frames, self.off_frames = min_area, on_frames, off_frames
        self.state, self._cnt = False, 0

    def update(self, fg_frac):
        hit = fg_frac >= self.min_area
        if hit != self.state:
            self._cnt += 1
            if self._cnt >= (self.on_frames if hit else self.off_frames):
                self.state, self._cnt = hit, 0
        else:
            self._cnt = 0
        return self.state


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def render(gray8, prob, mask, temps, rng, state, stats, thr, scale=2, recording=False):
    """Notebook-style figure: Input | Pred (cnn_mamba) | Overlay, white titles strip on top."""
    H, W = gray8.shape
    inp = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    pred = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)       # white on black
    over = inp.copy()
    m = mask.astype(bool)
    over[m] = (0.4 * over[m] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, cnts, -1, (0, 255, 255), 1)
    gap, top = 16, 40
    pw, ph = W * scale, H * scale
    canvas = np.full((top + ph + 14, 3 * pw + 4 * gap, 3), 255, np.uint8)
    for k, (img, title) in enumerate(((inp, "Input"), (pred, "Pred (cnn_mamba)"),
                                      (over, f"Overlay  (p >= {thr:.2f})"))):
        x0 = gap + k * (pw + gap)
        canvas[top:top + ph, x0:x0 + pw] = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_NEAREST
                                                      if k == 1 else cv2.INTER_LINEAR)
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(canvas, title, (x0 + (pw - tw) // 2, top - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"MLX90640 32x24 -> {W}x{H}   {temps.min():.1f}..{temps.max():.1f} C",
                (gap + 6, top + ph - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    bar = np.zeros((44, canvas.shape[1], 3), np.uint8)
    cv2.rectangle(bar, (0, 0), (262, 44), (0, 0, 255) if state else (0, 200, 0), -1)
    cv2.putText(bar, "CHANGE DETECTED" if state else "CLEAR", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(bar, stats, (278, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    if recording:
        cv2.circle(bar, (canvas.shape[1] - 20, 22), 9, (0, 0, 255), -1)
    return np.vstack([canvas, bar])


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="auto",
                    help="notebook checkpoint (.pt/.pth); 'auto' = newest file in weights/; "
                         "'none' = random weights (pipeline test only)")
    ap.add_argument("--source", default="serial",
                    help="serial[:PORT] | tcp:HOST[:PORT] | sim | replay:FILE.npz")
    ap.add_argument("--device", default="auto", help="auto | mps | cpu")
    ap.add_argument("--no-metal", action="store_true", help="use the chunked PyTorch scan instead of the Metal kernel")
    ap.add_argument("--threshold", type=float, default=None, help="pixel threshold (default: checkpoint's, 0.9)")
    ap.add_argument("--min-blob", type=float, default=0.03,
                    help="alarm when the largest foreground blob covers this fraction of the frame "
                         "(0.03 separated a person from sensor noise in a real MLX90640 recording)")
    ap.add_argument("--speck", type=float, default=0.005,
                    help="drop mask blobs smaller than this fraction (MLX90640 noise specks)")
    ap.add_argument("--norm", choices=["auto", "fixed"], default="auto")
    ap.add_argument("--tmin", type=float, default=18.0)
    ap.add_argument("--tmax", type=float, default=36.0)
    ap.add_argument("--min-span", type=float, default=8.0, help="auto-norm minimum degC span")
    ap.add_argument("--temporal", type=float, default=0.5,
                    help="frame averaging 0..0.9 (0 = off; higher = less noise, more motion lag)")
    ap.add_argument("--no-spatial", action="store_true", help="disable the anti-checkerboard 3x3 filter")
    ap.add_argument("--gate", type=float, default=1.0,
                    help="with a background: keep mask pixels only where the scene is >= this many "
                         "degC above room level (0 = off)")
    ap.add_argument("--bg-at-start", action="store_true",
                    help="capture the background from the first ~2 s (scene must be empty)")
    ap.add_argument("--flip", choices=["none", "h", "v", "hv"], default="none")
    ap.add_argument("--scale", type=int, default=2, help="display zoom")
    ap.add_argument("--record", action="store_true", help="start recording immediately")
    ap.add_argument("--headless", action="store_true", help="no window; print stats")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = run forever)")
    ap.add_argument("--save-every", type=float, default=0,
                    help="headless: save a rendered snapshot to recordings/ every N seconds")
    args = ap.parse_args()

    if args.no_metal:
        mscp.USE_METAL = False
    dev = mscp.pick_device(args.device)
    weights = None if args.weights.lower() == "none" else args.weights
    if weights == "auto":
        import glob
        found = sorted(glob.glob(os.path.join(HERE, "weights", "*.pt*")), key=os.path.getmtime)
        weights = found[-1] if found else os.path.join(HERE, "weights", "<missing>.pt")
    if weights and not os.path.exists(weights):
        sys.exit(f"Checkpoint not found: {weights}\n"
                 "Copy best__cnn_mamba_balanced.pt from Google Drive (MyDrive/mscp_mamba_balanced/) "
                 "into hardware/host/weights/, or pass --weights none to test the pipeline.")
    net, cfg = mscp.load_model(weights, dev)
    thr = args.threshold if args.threshold is not None else float(cfg.get("threshold", 0.9))
    H, W = cfg["img_size"]
    scan = "Metal kernel" if (dev.type == "mps" and mscp.USE_METAL and mscp.metal_available()) else "chunked PyTorch"
    print(f"model: {cfg['arch']} | weights: {weights or 'RANDOM (no checkpoint)'} | device: {dev} | "
          f"scan: {scan} | input {H}x{W} | threshold {thr}")

    with torch.inference_mode():                    # warm-up (Metal shader compile, MPS graphs)
        for _ in range(2):
            net(torch.zeros(1, 3, H, W, device=dev))
        if dev.type == "mps":
            torch.mps.synchronize()

    pre = Preprocessor((H, W), args.norm, args.tmin, args.tmax, args.min_span, flip=args.flip,
                       spatial=not args.no_spatial, temporal=args.temporal)
    det = Detector(args.min_blob)
    if args.bg_at_start:
        pre.start_background()
    src = open_source(args.source).start()
    print(f"source: {args.source}  (waiting for frames...)")

    rec, recording = [], args.record
    t_inf, t_lat, t_frames = deque(maxlen=30), deque(maxlen=30), deque(maxlen=30)
    n, last_print, win = 0, time.perf_counter(), "MSCP-Mamba thermal change detection"
    last_save = 0.0
    os.makedirs(os.path.join(HERE, "recordings"), exist_ok=True)

    def save_recording():
        if not rec:
            return
        path = os.path.join(HERE, "recordings", time.strftime("rec_%Y%m%d_%H%M%S.npz"))
        np.savez_compressed(path, temps=np.stack([r[0] for r in rec]),
                            t_ms=np.array([r[1] for r in rec], np.int64))
        print(f"saved {len(rec)} frames -> {path}")
        rec.clear()

    try:
        while True:
            fr = src.get(timeout=3.0)
            if fr is None:
                print("no frames for 3 s - is the ESP32 streaming? (LED blue = waiting for host,"
                      " purple blink = sensor not found)")
                src.send(b"P")
                continue
            if recording:
                rec.append((fr.temps, fr.t_ms))
            x, gray8, temps, rng = pre(fr.temps)

            t0 = time.perf_counter()
            with torch.inference_mode():
                prob = torch.sigmoid(net(x.to(dev)))[0, 0].float().cpu().numpy()   # .cpu() syncs
            t1 = time.perf_counter()
            raw = prob >= thr
            if pre.delta is not None and args.gate > 0:
                raw &= pre.delta >= args.gate       # thermal confirmation: must be warmer than the room
            mask, fg, blob = clean_mask(raw, args.speck)
            state = det.update(blob)
            src.send(b"D" if state else b"C")

            n += 1
            t_inf.append(t1 - t0)
            t_lat.append(t1 - fr.recv_time)
            t_frames.append(t1)
            fps = (len(t_frames) - 1) / (t_frames[-1] - t_frames[0]) if len(t_frames) > 1 else 0.0
            stats = (f"{fps:4.1f} fps | infer {1000 * np.mean(t_inf):3.0f} ms | "
                     f"latency {1000 * np.mean(t_lat):3.0f} ms | fg {100 * fg:4.1f}% | "
                     f"blob {100 * blob:4.1f}% | "
                     f"{'BG: CAPTURING' if pre.calibrating else ('BG on' if pre.bg is not None else 'BG off (b)')} | "
                     f"dropped {src.dropped}")

            if args.headless:
                if time.perf_counter() - last_print > 1.0:
                    print(f"[{n:5d}] {'DETECTED' if state else 'clear   '} | {stats}", flush=True)
                    last_print = time.perf_counter()
                if args.save_every and time.perf_counter() - last_save >= args.save_every:
                    last_save = time.perf_counter()
                    p = os.path.join(HERE, "recordings", f"live_{n:05d}.png")
                    cv2.imwrite(p, render(gray8, prob, mask, temps, rng, state, stats, thr, 2))
            else:
                cv2.imshow(win, render(gray8, prob, mask, temps, rng, state, stats, thr,
                                       args.scale, recording))
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                elif k == ord("r"):
                    recording = not recording
                    print("recording ON" if recording else "recording OFF")
                    if not recording:
                        save_recording()
                elif k == ord("s"):
                    p = os.path.join(HERE, "recordings", time.strftime("snap_%Y%m%d_%H%M%S.png"))
                    cv2.imwrite(p, render(gray8, prob, mask, temps, rng, state, stats, thr, 3))
                    print("snapshot ->", p)
                elif k == ord("f"):
                    order = ["none", "h", "v", "hv"]
                    pre.flip = order[(order.index(pre.flip) + 1) % 4]
                    pre.clear_background()
                    print("flip:", pre.flip, "(background cleared)")
                elif k == ord("b"):
                    pre.start_background()
                    print("capturing background - keep the scene empty for ~2 s ...")
                elif k == ord("x"):
                    pre.clear_background(); print("background cleared")
                elif k == ord("n"):
                    pre.norm = "fixed" if pre.norm == "auto" else "auto"
                    print("normalisation:", pre.norm)
                elif k == ord("["):
                    thr = max(0.05, round(thr - 0.05, 2)); print("threshold", thr)
                elif k == ord("]"):
                    thr = min(0.99, round(thr + 0.05, 2)); print("threshold", thr)
                elif chr(k) in "12486":
                    src.send(chr(k).encode()); print("sensor refresh ->", chr(k))
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        save_recording()
        src.close()
        cv2.destroyAllWindows()
        print(f"processed {n} frames | received {src.received} | dropped (newer frame arrived) {src.dropped}")


if __name__ == "__main__":
    main()
