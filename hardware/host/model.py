"""MSCP-Mamba (cnn_mamba, balanced) -- inference-only copy of the architecture in
`mscp_mamba_balanced_cdnet2014.ipynb`.

Module names / shapes are IDENTICAL to the notebook so its checkpoints load with
`load_state_dict(strict=True)`. Differences are inference-only and numerically
equivalent:
  * no grad-checkpointing / AMP / CUDA mamba kernels (runs on CPU or Apple MPS)
  * SS2D scans its 4 directions in ONE batched call (the 4 directions share
    x_proj / dt_proj / A / D, so stacking them on the batch dim is exact) --
    4x fewer sequential steps, which is what dominates latency on MPS.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Basic layers (notebook section 6)
# --------------------------------------------------------------------------- #
class ContrastNorm(nn.Module):
    def __init__(self, ch, eps=1e-5):
        super().__init__()
        self.norm = nn.InstanceNorm2d(ch, eps=eps, affine=False)

    def forward(self, x):
        return self.norm(x)


class SpatialDropout(nn.Module):
    def __init__(self, p=0.25):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        return F.dropout2d(x, p=self.p, training=True)


# --------------------------------------------------------------------------- #
# Encoder (section 7)
# --------------------------------------------------------------------------- #
class SparseModule(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, n_conv):
        super().__init__()
        layers, prev = [], in_ch
        for _ in range(n_conv):
            layers += [nn.Conv2d(prev, mid_ch, 3, padding=1), ContrastNorm(mid_ch), nn.ReLU(inplace=True)]
            prev = mid_ch
        layers += [nn.Conv2d(prev, out_ch, 1)]
        layers += [ContrastNorm(out_ch), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AtrousModule(nn.Module):
    def __init__(self, in_ch, out_ch, rate):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=rate, dilation=rate)
        self.cn, self.act = ContrastNorm(out_ch), nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.cn(self.conv(x)))


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch=64, n_sparse=2):
        super().__init__()
        self.m1 = SparseModule(in_ch, mid_ch, out_ch, n_sparse)
        self.a2 = AtrousModule(in_ch, out_ch, 2)
        self.a4 = AtrousModule(in_ch, out_ch, 4)
        self.a8 = AtrousModule(in_ch, out_ch, 8)
        self.a16 = AtrousModule(in_ch, out_ch, 16)

    def forward(self, x):
        return self.m1(x) + self.a2(x) + self.a4(x) + self.a8(x) + self.a16(x)


class Encoder(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        self.b1 = EncoderBlock(in_ch, 64, base, 2)
        self.b2 = EncoderBlock(base, 128, base, 2)
        self.b3 = EncoderBlock(base, 256, base, 3)
        self.b4 = EncoderBlock(base, 512, base, 3)
        self.sd, self.pool = SpatialDropout(0.25), nn.MaxPool2d(2, 2)
        self.proj1 = nn.Conv2d(base, base, 1)
        self.fuse = nn.Conv2d(base * 4, 192, 1)

    def forward(self, x):
        o1 = self.b1(x)
        low_full = o1
        z = self.pool(self.sd(o1)); o2 = self.b2(z) + z
        z = self.pool(self.sd(o2)); o3 = self.b3(z) + z
        z = self.sd(o3);            o4 = self.b4(z) + z
        o1p = self.proj1(o1)
        size = o4.shape[2:]
        o1a = F.interpolate(o1p, size=size, mode="bilinear", align_corners=False)
        o2a = F.interpolate(o2, size=size, mode="bilinear", align_corners=False)
        Ff = self.fuse(torch.cat([o1a, o2a, o3, o4], dim=1))
        return Ff, low_full


# --------------------------------------------------------------------------- #
# MSCP block (section 8)
# --------------------------------------------------------------------------- #
class MSCPBlock(nn.Module):
    def __init__(self, in_ch=192, out_ch=240, mid=64):
        super().__init__()
        self.M_conv = nn.Conv2d(in_ch, mid, 1)
        self.cv1 = nn.Conv2d(in_ch, mid, 3, padding=1)
        self.cv2 = nn.Conv2d(in_ch, mid, 3, padding=4, dilation=4)
        self.cv3 = nn.Conv2d(in_ch, mid, 3, padding=8, dilation=8)
        self.cv4 = nn.Conv2d(in_ch, mid, 3, padding=16, dilation=16)
        self.avg = nn.AvgPool2d(3, stride=1, padding=1)
        self.proj = nn.Conv2d(mid * 5, out_ch, 1)
        self.cn, self.act, self.sd = ContrastNorm(out_ch), nn.ReLU(inplace=True), SpatialDropout(0.25)

    def _contrast(self, conv, x):
        cv = conv(x)
        return F.relu(cv - self.avg(cv))

    def forward(self, x):
        M = F.relu(self.M_conv(F.max_pool2d(x, 2)))
        M = F.interpolate(M, size=x.shape[2:], mode="bilinear", align_corners=False)
        x1, x2, x3, x4 = (self._contrast(self.cv1, x), self._contrast(self.cv2, x),
                          self._contrast(self.cv3, x), self._contrast(self.cv4, x))
        y = self.proj(torch.cat([M, x1, x2, x3, x4], dim=1))
        return self.sd(self.act(self.cn(y)))


# --------------------------------------------------------------------------- #
# Selective scan (section 8b) -- chunked, pure PyTorch
# --------------------------------------------------------------------------- #
def chunked_selective_scan(u, delta, A, B, C, D=None, chunk=64):
    """u, delta [b,d,l] | A [d,n] | B, C [b,n,l] -> y [b,d,l] (delta already softplus'ed)."""
    b, d, l = u.shape
    n = A.shape[1]
    k = max(1, min(chunk, l))
    t = (l + k - 1) // k
    pad = t * k - l
    if pad:
        u = F.pad(u, (0, pad))
        delta = F.pad(delta, (0, pad))
        B = F.pad(B, (0, pad))
        C = F.pad(C, (0, pad))
    u_ = u.view(b, d, t, k)
    dt_ = delta.view(b, d, t, k)
    B_ = B.view(b, n, t, k).permute(0, 2, 3, 1).unsqueeze(1)
    C_ = C.view(b, n, t, k).permute(0, 2, 3, 1).unsqueeze(1)
    dtA = dt_.unsqueeze(-1) * A.view(1, d, 1, 1, n)
    dA = torch.exp(dtA)
    dBu = dt_.unsqueeze(-1) * u_.unsqueeze(-1) * B_
    # pass 1: chunk-local scan from zero state
    h = dA.new_zeros(b, d, t, n)
    yloc = []
    for i in range(k):
        h = dA[:, :, :, i] * h + dBu[:, :, :, i]
        yloc.append((h * C_[:, :, :, i]).sum(-1))
    # carries between chunks
    a = torch.exp(dtA.sum(dim=3))
    h_end = h
    cs, cs_list = a.new_zeros(b, d, n), []
    for j in range(t):
        cs_list.append(cs)
        cs = a[:, :, j] * cs + h_end[:, :, j]
    cs = torch.stack(cs_list, dim=2)
    # pass 2: add contribution of each chunk's start state
    h2 = cs
    for i in range(k):
        h2 = dA[:, :, :, i] * h2
        yloc[i] = yloc[i] + (h2 * C_[:, :, :, i]).sum(-1)
    y = torch.stack(yloc, dim=-1).view(b, d, t * k)[..., :l]
    if D is not None:
        y = y + u[..., :l] * D.view(1, -1, 1)
    return y


def selective_scan(u, delta, A, B, C, D=None, chunk=64):
    u, delta, A, B, C = (x.float() for x in (u, delta, A, B, C))
    delta = F.softplus(delta)
    return chunked_selective_scan(u, delta, A, B, C, None if D is None else D.float(), chunk)


# --------------------------------------------------------------------------- #
# Apple-GPU (Metal) selective scan -- same math as the official mamba CUDA
# kernel: one GPU thread per (sequence, channel), SSM state kept in registers,
# sequential over L. ~30x faster than the chunked fallback on an M-series GPU.
# --------------------------------------------------------------------------- #
_METAL_SRC = r"""
#include <metal_stdlib>
using namespace metal;
// layouts: u, delta, y: [B, L, D]   Bm, Cm: [B, L, 16]   A: [D, 16]   Dv: [D]
kernel void sscan(device const float* u     [[buffer(0)]],
                  device const float* delta [[buffer(1)]],
                  device const float* A     [[buffer(2)]],
                  device const float* Bm    [[buffer(3)]],
                  device const float* Cm    [[buffer(4)]],
                  device const float* Dv    [[buffer(5)]],
                  device float*       y     [[buffer(6)]],
                  constant long& Bsz [[buffer(7)]],
                  constant long& L   [[buffer(8)]],
                  constant long& D   [[buffer(9)]],
                  uint tid [[thread_position_in_grid]]) {
    const int N = 16;
    long b = tid / D, d = tid % D;
    if (b >= Bsz) return;
    float a[N], h[N];
    for (int n = 0; n < N; ++n) { a[n] = A[d * N + n]; h[n] = 0.0f; }
    float dd = Dv[d];
    long base = b * L * D + d, bn = b * L * N;
    for (long l = 0; l < L; ++l) {
        float dt = delta[base + l * D], x = u[base + l * D];
        float dtx = dt * x, acc = 0.0f;
        device const float* Bl = Bm + bn + l * N;
        device const float* Cl = Cm + bn + l * N;
        for (int n = 0; n < N; ++n) {
            h[n] = exp(dt * a[n]) * h[n] + dtx * Bl[n];
            acc += h[n] * Cl[n];
        }
        y[base + l * D] = acc + x * dd;
    }
}
"""
_metal_lib = None
USE_METAL = True          # set False (or --no-metal) to force the chunked fallback


def _get_metal():
    global _metal_lib
    if _metal_lib is None:
        _metal_lib = torch.mps.compile_shader(_METAL_SRC)
    return _metal_lib


def metal_scan_bld(u, delta, A, B, C, D):
    """u, delta [b,l,d] (delta already softplus'ed) | A [d,16] | B, C [b,l,16] | D [d]
       -> y [b,l,d]. All tensors on MPS."""
    b, l, d = u.shape
    y = torch.empty((b, l, d), device=u.device, dtype=torch.float32)   # contiguous [b,l,d]
    _get_metal().sscan(u.contiguous(), delta.contiguous(), A.contiguous(),
                       B.contiguous(), C.contiguous(), D.contiguous(), y,
                       b, l, d, threads=b * d)
    return y


def metal_available():
    try:
        return torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader") \
            and _get_metal() is not None
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# SS2D + VSS block (sections 8c, 8d)
# --------------------------------------------------------------------------- #
class SS2D(nn.Module):
    def __init__(self, d_model, expand=2, d_state=16, chunk=64):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = max(1, self.d_inner // 16)
        self.chunk = chunk
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, 3, padding=1,
                                groups=self.d_inner, bias=True)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.05) - math.log(1e-3)) + math.log(1e-3))
        with torch.no_grad():
            self.dt_proj.bias.copy_(torch.log(torch.expm1(dt)))

    def _scan1d(self, seq):
        dr, n = self.dt_rank, self.d_state
        if USE_METAL and seq.device.type == "mps" and n == 16:
            xt = seq.transpose(1, 2).float()                       # [b,l,d]
            x_dbl = self.x_proj(xt)
            delta = F.softplus(self.dt_proj(x_dbl[..., :dr]).float())
            A = -torch.exp(self.A_log.float())
            y = metal_scan_bld(xt, delta, A, x_dbl[..., dr:dr + n].float(),
                               x_dbl[..., dr + n:].float(), self.D.float())
            return y.transpose(1, 2)
        x_dbl = self.x_proj(seq.transpose(1, 2))
        delta = self.dt_proj(x_dbl[..., :dr]).transpose(1, 2)
        B = x_dbl[..., dr:dr + n].transpose(1, 2)
        C = x_dbl[..., dr + n:].transpose(1, 2)
        A = -torch.exp(self.A_log.float())
        return selective_scan(seq, delta, A, B, C, self.D.float(), self.chunk)

    def forward(self, x):                                   # x [b,h,w,d]
        b, h, w, _ = x.shape
        xz = self.in_proj(x)
        xv, z = xz.chunk(2, dim=-1)
        xv = self.act(self.conv2d(xv.permute(0, 3, 1, 2).contiguous()))   # [b,di,h,w]
        di = xv.shape[1]
        # 4 scan orders stacked on the batch dim -> one scan call
        rm = xv.flatten(2)
        cm = xv.transpose(2, 3).flatten(2)
        seqs = torch.cat([rm, rm.flip(-1), cm, cm.flip(-1)], dim=0)        # [4b,di,l]
        ys = self._scan1d(seqs).view(4, b, di, h * w)
        y = (ys[0].reshape(b, di, h, w)
             + ys[1].flip(-1).reshape(b, di, h, w)
             + ys[2].reshape(b, di, w, h).transpose(2, 3)
             + ys[3].flip(-1).reshape(b, di, w, h).transpose(2, 3)) / 4.0
        y = y * self.act(z.permute(0, 3, 1, 2))
        return self.out_proj(y.permute(0, 2, 3, 1))


class VSSBlock(nn.Module):
    def __init__(self, dim, expand=2, d_state=16, chunk=64, drop=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(dim, expand, d_state, chunk)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        y = self.norm(x.permute(0, 2, 3, 1))
        y = self.ss2d(y).permute(0, 3, 1, 2)
        return x + self.drop(y)


# --------------------------------------------------------------------------- #
# Decoder (section 9) + full model (section 10)
# --------------------------------------------------------------------------- #
class FusionLayer(nn.Module):
    def __init__(self, dec_ch, low_ch, out_ch=64):
        super().__init__()
        self.low_conv = nn.Conv2d(low_ch, out_ch, 1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

    def forward(self, dec_out, low):
        v = self.gmp(self.low_conv(low))
        v = F.interpolate(v, size=dec_out.shape[2:], mode="bilinear", align_corners=False)
        return (v * dec_out) + dec_out


class Decoder(nn.Module):
    def __init__(self, in_ch=240, mid=64):
        super().__init__()

        def blk(cin):
            return nn.Sequential(nn.Conv2d(cin, mid, 3, padding=1),
                                 ContrastNorm(mid), nn.ReLU(inplace=True))
        self.b1, self.b2, self.b3 = blk(in_ch), blk(mid), blk(mid)
        self.fuse1 = FusionLayer(mid, 64, mid)
        self.fuse2 = FusionLayer(mid, 64, mid)
        self.head = nn.Conv2d(mid, 1, 1)

    def forward(self, y, low_full):
        d = self.b1(y); d = self.fuse1(d, low_full)
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.b2(d); d = self.fuse2(d, low_full)
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.b3(d)
        return self.head(d)


class MSCPNet(nn.Module):
    def __init__(self, in_ch=3, arch="cnn_mamba", n_vss=2,
                 vss_expand=2, vss_d_state=16, scan_chunk=64):
        super().__init__()
        self.encoder = Encoder(in_ch)
        self.mscp = MSCPBlock(192, 240)
        self.arch = arch
        if arch == "cnn_mamba":
            self.vss = nn.ModuleList([VSSBlock(240, vss_expand, vss_d_state, scan_chunk)
                                      for _ in range(n_vss)])
        elif arch == "cnn":
            self.vss = None
        else:
            raise ValueError(arch)
        self.decoder = Decoder(240)

    def forward(self, x):
        Ff, low_full = self.encoder(x)
        y = self.mscp(Ff)
        if self.vss is not None:
            for blk in self.vss:
                y = blk(y)
        return self.decoder(y, low_full)


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(weights, device, scan_chunk=None):
    """Build MSCPNet from a notebook checkpoint. Accepts any of:
       best__cnn_mamba_balanced.pt      {"state_dict", "config", ...}
       mscp_mamba_balanced_cdnet2014.pth {"state_dict", "config", "arch"}
       last__cnn_mamba_balanced.pt      {"model", "config", ...}
       weights=None -> random init (pipeline testing only)."""
    cfg = dict(arch="cnn_mamba", n_vss=2, vss_expand=2, vss_d_state=16,
               scan_chunk=64, in_channels=3, img_size=(240, 320), threshold=0.9)
    state = None
    if weights:
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "state_dict" in ck:
            state = ck["state_dict"]
        elif isinstance(ck, dict) and "model" in ck:
            state = ck["model"]
        else:
            state = ck
        if isinstance(ck, dict) and isinstance(ck.get("config"), dict):
            cfg.update({k: v for k, v in ck["config"].items() if k in cfg})
        state = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in state.items()}
    if scan_chunk:
        cfg["scan_chunk"] = scan_chunk
    model = MSCPNet(cfg["in_channels"], cfg["arch"], cfg["n_vss"],
                    cfg["vss_expand"], cfg["vss_d_state"], cfg["scan_chunk"])
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model.to(device).eval(), cfg
