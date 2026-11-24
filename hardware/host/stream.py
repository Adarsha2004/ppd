"""Frame sources for the real-time app.

All sources yield `ThermalFrame`s (24x32 float32 degC) from a background thread
and keep only the NEWEST frame -- if inference is slower than the sensor, old
frames are dropped instead of building up latency.

  SerialSource  ESP32-S3 over USB            (--source serial[:PORT])
  TcpSource     ESP32-S3 over WiFi           (--source tcp:HOST[:PORT])
  ReplaySource  a recording made with 'r'    (--source replay:FILE.npz)
  SimSource     synthetic people walking     (--source sim)
"""
import binascii
import glob
import socket
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np

MAGIC = b"\xA5\x5AMX"
ROWS, COLS = 24, 32
HDR = struct.Struct("<IIhH")                 # seq, t_ms, ta (centi-degC), flags
PKT_LEN = len(MAGIC) + HDR.size + ROWS * COLS * 2 + 2   # 1554


@dataclass
class ThermalFrame:
    temps: np.ndarray      # (24, 32) float32 degC
    seq: int
    t_ms: int              # device clock
    ta: float              # sensor ambient degC
    recv_time: float       # host time.perf_counter() at arrival


class PacketParser:
    """Byte-stream -> packets. Resyncs on magic + CRC; prints '# ...' log lines."""

    def __init__(self, log=print):
        self.buf = bytearray()
        self.log = log
        self.crc_errors = 0

    def feed(self, data):
        self.buf += data
        out = []
        while True:
            i = self.buf.find(MAGIC)
            if i < 0:                    # flush complete text lines, keep the rest
                cut = self.buf.rfind(b"\n") + 1                  # (may hold a partial magic)
                if len(self.buf) - cut > 2048:
                    cut = len(self.buf) - 3
                self._logs(self.buf[:cut]); del self.buf[:cut]
                return out
            if i > 0:
                self._logs(self.buf[:i]); del self.buf[:i]
            if len(self.buf) < PKT_LEN:
                return out
            body = bytes(self.buf[4:PKT_LEN - 2])
            crc, = struct.unpack_from("<H", self.buf, PKT_LEN - 2)
            if binascii.crc_hqx(body, 0xFFFF) != crc:
                self.crc_errors += 1
                del self.buf[:4]                                 # skip this magic
                continue
            seq, t_ms, ta, _flags = HDR.unpack_from(body, 0)
            temps = np.frombuffer(body, "<i2", ROWS * COLS, HDR.size)
            temps = (temps.astype(np.float32) / 100.0).reshape(ROWS, COLS)
            out.append(ThermalFrame(temps, seq, t_ms, ta / 100.0, time.perf_counter()))
            del self.buf[:PKT_LEN]

    def _logs(self, junk):
        for line in bytes(junk).split(b"\n"):
            j = line.find(b"# ")          # may follow binary leftovers of a bad packet
            if j >= 0:
                self.log("[esp32] " + line[j:].strip().decode(errors="replace"))


class _Source:
    """Latest-frame mailbox fed by a daemon thread."""

    def __init__(self):
        self._cv = threading.Condition()
        self._latest = None
        self._stop = False
        self.received = 0
        self.dropped = 0
        self.error = None
        self._thread = threading.Thread(target=self._run_guarded, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _run_guarded(self):
        try:
            self._run()
        except Exception as e:           # surface reader errors to the main loop
            self.error = e
            with self._cv:
                self._cv.notify_all()

    def _publish(self, frame):
        with self._cv:
            if self._latest is not None:
                self.dropped += 1
            self._latest = frame
            self.received += 1
            self._cv.notify_all()

    def get(self, timeout=2.0):
        """Newest unseen frame, or None on timeout."""
        with self._cv:
            if self._latest is None and self.error is None:
                self._cv.wait(timeout)
            if self.error is not None:
                raise RuntimeError(f"frame source failed: {self.error}") from self.error
            f, self._latest = self._latest, None
            return f

    def send(self, data: bytes):
        pass

    def close(self):
        self._stop = True

    def _run(self):
        raise NotImplementedError


class SerialSource(_Source):
    def __init__(self, port="auto", baud=921600):
        super().__init__()
        import serial                                # pyserial
        if port in (None, "", "auto"):
            port = self.autodetect()
        self.port = port
        self.ser = serial.Serial(port, baud, timeout=0.05)
        self.ser.reset_input_buffer()
        self.parser = PacketParser()
        self._wlock = threading.Lock()

    @staticmethod
    def autodetect():
        cands = sorted(glob.glob("/dev/cu.usbmodem*")) + sorted(glob.glob("/dev/cu.usbserial*")) \
            + sorted(glob.glob("/dev/cu.wchusbserial*")) + sorted(glob.glob("/dev/cu.SLAB*")) \
            + sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*"))
        if not cands:
            raise RuntimeError("No ESP32 serial port found. Plug the board in (USB port), "
                               "or pass --source serial:/dev/cu.XXXX")
        return cands[0]

    def _run(self):
        while not self._stop:
            data = self.ser.read(max(1, min(self.ser.in_waiting, 8192)))
            if data:
                for f in self.parser.feed(data):
                    self._publish(f)

    def send(self, data):
        with self._wlock:
            try:
                self.ser.write(data)
            except Exception:
                pass

    def close(self):
        super().close()
        time.sleep(0.1)
        self.ser.close()


class TcpSource(_Source):
    def __init__(self, host, port=3333):
        super().__init__()
        self.addr = (host, port)
        self.sock = socket.create_connection(self.addr, timeout=5)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(0.5)
        self.parser = PacketParser()

    def _run(self):
        while not self._stop:
            try:
                data = self.sock.recv(8192)
            except socket.timeout:
                continue
            if not data:
                raise ConnectionError("ESP32 closed the TCP connection")
            for f in self.parser.feed(data):
                self._publish(f)

    def send(self, data):
        try:
            self.sock.sendall(data)
        except OSError:
            pass

    def close(self):
        super().close()
        self.sock.close()


class ReplaySource(_Source):
    """Plays back an .npz recording (temps [T,24,32], t_ms [T]) at its recorded rate."""

    def __init__(self, path, loop=True, speed=1.0):
        super().__init__()
        d = np.load(path)
        self.temps, self.t_ms = d["temps"].astype(np.float32), d["t_ms"].astype(np.int64)
        self.loop, self.speed = loop, speed

    def _run(self):
        seq = 0
        while not self._stop:
            t0 = time.perf_counter()
            base = int(self.t_ms[0])
            for temps, tm in zip(self.temps, self.t_ms):
                wait = (tm - base) / 1000.0 / self.speed - (time.perf_counter() - t0)
                if wait > 0:
                    time.sleep(wait)
                if self._stop:
                    return
                self._publish(ThermalFrame(temps.copy(), seq, int(tm), float("nan"),
                                           time.perf_counter()))
                seq += 1
            if not self.loop:
                return


class SimSource(_Source):
    """Synthetic MLX90640: 22 degC room, sensor noise, warm 'people' walking around.
    For testing the pipeline without hardware (not for judging model accuracy)."""

    def __init__(self, fps=8.0, n_people=2, seed=0):
        super().__init__()
        self.fps, self.n = fps, n_people
        self.rng = np.random.default_rng(seed)

    def _run(self):
        yy, xx = np.mgrid[0:ROWS, 0:COLS].astype(np.float32)
        bg = 22.0 + 1.5 * (yy / ROWS) + 0.8 * np.sin(xx / 6.0)        # mild gradient
        bg[2:6, 24:30] += 6.0                                          # static warm object (radiator)
        seq, t0 = 0, time.perf_counter()
        while not self._stop:
            t = seq / self.fps
            img = bg + self.rng.normal(0, 0.25, bg.shape)
            for k in range(self.n):
                if (t + 7 * k) % 20 > 14:                              # people leave the scene
                    continue
                cx = COLS / 2 + (COLS / 2 + 4) * np.sin(0.25 * t + 2.5 * k)
                cy = ROWS * 0.55 + 3 * np.sin(0.4 * t + k)
                body = np.exp(-(((xx - cx) / 2.2) ** 2 + ((yy - cy) / 5.0) ** 2))
                head = np.exp(-(((xx - cx) / 1.4) ** 2 + ((yy - (cy - 6.5)) / 1.4) ** 2))
                img = np.maximum(img, 22 + 11 * body) + 3 * head
            self._publish(ThermalFrame(img.astype(np.float32), seq, int(t * 1000), 25.0,
                                       time.perf_counter()))
            seq += 1
            nxt = t0 + seq / self.fps
            time.sleep(max(0.0, nxt - time.perf_counter()))


def open_source(spec):
    """'sim' | 'serial' | 'serial:/dev/cu.usbmodem1101' | 'tcp:thermal-cam.local[:3333]'
       | 'replay:recordings/rec.npz'"""
    kind, _, arg = spec.partition(":")
    if kind == "sim":
        return SimSource()
    if kind == "serial":
        return SerialSource(arg or "auto")
    if kind == "tcp":
        host, _, port = arg.partition(":")
        return TcpSource(host or "thermal-cam.local", int(port or 3333))
    if kind == "replay":
        return ReplaySource(arg)
    raise ValueError(f"unknown --source {spec!r}")
