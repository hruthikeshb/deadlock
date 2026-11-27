"""Independent 70-point eval of monsoon_net_best.pt.

Same windows, baselines, shocks, and scoring as the GRAST independent eval.
Architecture and rollout copied from Downloads/kaggle_train.ipynb (MonsoonNet).
Does not import training/evaluate.py or tests/benchmark_metrics.py.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import DatasetReader, Transform  # data only
from src.features.geo import build_geo_features

# ---------------------------------------------------------------------------
# MonsoonNet (from kaggle_train.ipynb)
# ---------------------------------------------------------------------------
N_CHANNELS = 6
N_GEO = 6
N_DOY = 2
RAIN_CH = 1
TCWV_CH = 5
GRID_H, GRID_W = 32, 36
C_IN = 3 * N_CHANNELS + N_GEO + N_DOY


class ReplicatePad(nn.Module):
    def __init__(self, pad: int = 1):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (self.pad,) * 4, mode="replicate")


def conv3x3(in_ch: int, out_ch: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, 3, padding=0, bias=False, **kwargs)


class ResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.pad = ReplicatePad(1)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = conv3x3(channels, channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = conv3x3(channels, channels)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(self.pad(h))
        h = F.silu(self.norm2(h))
        h = self.drop(h)
        h = self.conv2(self.pad(h))
        return x + h


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pad = ReplicatePad(1)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=0, bias=False)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(self.conv(self.pad(x))))


class UpBlock(nn.Module):
    def __init__(self, coarse_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.pad = ReplicatePad(1)
        self.conv = conv3x3(coarse_ch + skip_ch, out_ch)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, coarse: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(coarse, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return F.silu(self.norm(self.conv(self.pad(torch.cat([up, skip], dim=1)))))


class ConvGRUCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, kernel: int = 3):
        super().__init__()
        self.pad = ReplicatePad(kernel // 2)
        self.reset_gate = nn.Conv2d(in_ch + hid_ch, hid_ch, kernel, padding=0, bias=True)
        self.update_gate = nn.Conv2d(in_ch + hid_ch, hid_ch, kernel, padding=0, bias=True)
        self.out_gate = nn.Conv2d(in_ch + hid_ch, hid_ch, kernel, padding=0, bias=True)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        xh = self.pad(torch.cat([x, h], dim=1))
        r = torch.sigmoid(self.reset_gate(xh))
        z = torch.sigmoid(self.update_gate(xh))
        rh = self.pad(torch.cat([x, r * h], dim=1))
        n = torch.tanh(self.out_gate(rh))
        return (1 - z) * h + z * n


class HistoryEncoder(nn.Module):
    def __init__(self, frame_ch: int, hid_ch: int):
        super().__init__()
        self.frame_proj = nn.Sequential(
            ReplicatePad(1), conv3x3(frame_ch, hid_ch),
            nn.GroupNorm(min(8, hid_ch), hid_ch), nn.SiLU(),
        )
        self.gru = ConvGRUCell(hid_ch, hid_ch)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = history.shape
        h = torch.zeros(B, self.gru.reset_gate.out_channels, H, W, device=history.device, dtype=history.dtype)
        for t in range(T):
            h = self.gru(self.frame_proj(history[:, t]), h)
        return h


class FiLMLayer(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, channels)
        self.beta = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(cond)[:, :, None, None]
        beta = self.beta(cond)[:, :, None, None]
        return x * (1 + gamma) + beta


class PrecipHead(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        mid = max(in_ch // 2, 8)
        self.occ = nn.Sequential(
            ReplicatePad(1), conv3x3(in_ch, mid), nn.SiLU(),
            ReplicatePad(1), conv3x3(mid, 1),
        )
        self.intensity = nn.Sequential(
            ReplicatePad(1), conv3x3(in_ch, mid), nn.SiLU(),
            ReplicatePad(1), conv3x3(mid, 1),
        )

    def forward(self, feat: torch.Tensor):
        occ_logit = self.occ(feat)
        occ_prob = torch.sigmoid(occ_logit)
        intensity = F.softplus(self.intensity(feat))
        return occ_prob * intensity, occ_logit, intensity


class MonsoonNet(nn.Module):
    def __init__(
        self,
        c_in: int = C_IN,
        base_ch: int = 48,
        n_fine_pre: int = 2,
        n_mid: int = 2,
        n_coarse: int = 3,
        n_fine_post: int = 2,
        dropout: float = 0.05,
        cond_dim: int = 16,
    ):
        super().__init__()
        ch1, ch2, ch3 = base_ch, base_ch * 2, base_ch * 4
        self.history_enc = HistoryEncoder(frame_ch=N_CHANNELS, hid_ch=ch1)
        self.stem = nn.Sequential(
            ReplicatePad(1), nn.Conv2d(ch1 + N_GEO + N_DOY, ch1, 3, padding=0, bias=False),
            nn.GroupNorm(min(8, ch1), ch1), nn.SiLU(),
        )
        self.fine_pre = nn.Sequential(*[ResBlock(ch1, dropout) for _ in range(n_fine_pre)])
        self.down1 = DownBlock(ch1, ch2)
        self.mid = nn.Sequential(*[ResBlock(ch2, dropout) for _ in range(n_mid)])
        self.down2 = DownBlock(ch2, ch3)
        self.coarse = nn.Sequential(*[ResBlock(ch3, dropout) for _ in range(n_coarse)])
        self.up2 = UpBlock(ch3, ch2, ch2)
        self.up1 = UpBlock(ch2, ch1, ch1)
        self.lead_emb = nn.Sequential(nn.Linear(1, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.doy_emb = nn.Sequential(nn.Linear(2, cond_dim), nn.SiLU())
        self.film = FiLMLayer(cond_dim * 2, ch1)
        self.fine_post = nn.Sequential(*[ResBlock(ch1, dropout) for _ in range(n_fine_post)])
        self.weather_head = nn.Sequential(
            ReplicatePad(1), nn.Conv2d(ch1, ch1 // 2, 3, padding=0, bias=False), nn.SiLU(),
            ReplicatePad(1), nn.Conv2d(ch1 // 2, N_CHANNELS - 1, 3, padding=0, bias=True),
        )
        self.precip_head = PrecipHead(ch1)
        self.res_gain = nn.Parameter(torch.full((N_CHANNELS,), 0.1))

    def forward(self, history, geo, doy_sin, doy_cos, lead):
        B, T, C, H, W = history.shape
        h_enc = self.history_enc(history)
        doy_map = torch.stack([doy_sin, doy_cos], dim=1)[:, :, None, None].expand(B, 2, H, W)
        x = self.stem(torch.cat([h_enc, geo, doy_map], dim=1))
        skip0 = self.fine_pre(x)
        skip1 = self.mid(self.down1(skip0))
        c = self.coarse(self.down2(skip1))
        u1 = self.up1(self.up2(c, skip1), skip0)
        cond = torch.cat([
            self.lead_emb(lead.float().unsqueeze(1) / 10.0),
            self.doy_emb(torch.stack([doy_sin, doy_cos], dim=1)),
        ], dim=1)
        feat = self.fine_post(self.film(u1, cond))
        weather_delta = self.weather_head(feat)
        rain_delta, occ_logit, intensity = self.precip_head(feat)
        delta = torch.cat([weather_delta[:, :1], rain_delta, weather_delta[:, 1:]], dim=1)
        return delta * self.res_gain.view(1, N_CHANNELS, 1, 1), occ_logit, intensity


@torch.no_grad()
def monsoon_rollout_phys(
    model: nn.Module,
    transform: Transform,
    geo_np: np.ndarray,
    init_states: np.ndarray,
    doy_sincos: np.ndarray,
    steps: int,
    perturbation: np.ndarray | None = None,
) -> np.ndarray:
    """Physical 10-step rollout matching kaggle_train.ipynb monsoon_rollout (no teacher forcing)."""
    states = init_states.copy()
    if perturbation is not None:
        states[-1] = states[-1] + perturbation

    rain_norm_min = float(-transform.mean[RAIN_CH] / transform.std[RAIN_CH])
    tcwv_norm_min = float(-transform.mean[TCWV_CH] / transform.std[TCWV_CH])

    hist_np = np.stack([transform.normalize(s) for s in states], axis=0)  # [3,6,H,W]
    hist = torch.from_numpy(hist_np[None]).float()  # [1,3,6,H,W]
    geo_t = torch.from_numpy(geo_np[None]).float()
    sin_v = torch.tensor([float(doy_sincos[0])])
    cos_v = torch.tensor([float(doy_sincos[1])])
    cd = math.cos(2.0 * math.pi / 365.0)
    sd = math.sin(2.0 * math.pi / 365.0)

    model.eval()
    outs = []
    for step in range(steps):
        lead_t = torch.full((1,), step + 1, dtype=torch.long)
        delta, _, _ = model(hist, geo_t, sin_v, cos_v, lead_t)
        z_next = hist[:, -1] + delta
        z_next = z_next.clone()
        z_next[:, RAIN_CH] = z_next[:, RAIN_CH].clamp(min=rain_norm_min)
        z_next[:, TCWV_CH] = z_next[:, TCWV_CH].clamp(min=tcwv_norm_min)
        x_next = transform.denormalize(z_next.squeeze(0).numpy())
        x_next[RAIN_CH] = np.maximum(x_next[RAIN_CH], 0.0)
        x_next[TCWV_CH] = np.maximum(x_next[TCWV_CH], 0.0)
        outs.append(x_next)
        hist = torch.cat([hist[:, 1:], torch.from_numpy(transform.normalize(x_next))[None, None]], dim=1)
        sin_v, cos_v = sin_v * cd + cos_v * sd, cos_v * cd - sin_v * sd
    return np.stack(outs, axis=0)


# ---------------------------------------------------------------------------
# Shared fair rubric (identical windows / shocks / point maps as GRAST eval)
# ---------------------------------------------------------------------------
CHANNELS = ["t2m", "tp", "u10", "v10", "msl", "tcwv"]
STD = np.array([13.46363353729248, 7.553342342376709, 2.8989806175231934,
                2.4269521236419678, 619.0198364257812, 20.49437141418457], dtype=np.float32)
LEADS = (0, 3, 7)
RAIN_THR = 10.0
EVENTS = {
    "monsoon_onset": ["2020-06-08", "2021-06-10"],
    "heavy_rain": ["2020-07-15", "2021-08-12"],
    "heatwave": ["2020-05-22", "2021-05-18"],
    "cyclone": ["2020-11-24", "2021-05-16"],
    "winter": ["2020-01-20", "2021-02-04"],
}

CKPT_CANDIDATES = [
    ROOT / "artifacts" / "monsoon_net_best.pt",
    Path(r"C:\Users\hruth\Downloads\monsoon_net_best.pt.zip"),
    Path(r"C:\Users\hruth\Downloads\monsoon_net_best.pt"),
]


def date_to_idx(dates, s):
    hits = np.where(dates == np.datetime64(s, "D"))[0]
    if len(hits) == 0:
        raise KeyError(s)
    return int(hits[0])


def doy_of(dates, idx):
    t = dates[idx]
    yr = t.astype("datetime64[Y]").astype("datetime64[D]")
    return int((t - yr).astype(int)) + 1


def sincos(doy):
    a = 2.0 * np.pi * doy / 365.0
    return np.array([np.sin(a), np.cos(a)], dtype=np.float32)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def csi(pred, truth, thr):
    p = pred >= thr
    t = truth >= thr
    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()
    den = tp + fp + fn
    return float(tp / den) if den else 1.0


def spatial_std(x):
    return float(np.std(x))


def highfreq_energy(x):
    f = np.fft.rfft2(x)
    p = np.abs(f) ** 2
    h, w = p.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy / max(h - 1, 1)) ** 2 + (xx / max(w - 1, 1)) ** 2)
    hi = p[r > 0.35].sum()
    tot = p.sum() + 1e-12
    return float(hi / tot)


def clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def error_points_from_mean_rmse(recs):
    """Calibrated error map used for the published GRAST 41/70 score.

    S is mean-over-channels of (1 - RMSE_model / RMSE_bestref), where each RMSE
    is first averaged across the 10 windows. Points:
        lead_weight * clip((S + 0.10) / 0.40, 0, 1)
    so a tie with the best baseline is not zero, a ~30% RMSE win is full marks.
    """
    lead_weights = {0: 8.0, 3: 12.0, 7: 10.0}
    out = {}
    beat_persist_d4 = []
    total = 0.0
    for lead, w in lead_weights.items():
        ch_s = []
        for c in range(6):
            rm = np.mean([rmse(r["pred"][lead, c], r["truth"][lead, c]) for r in recs])
            rp = np.mean([rmse(r["persist"][lead, c], r["truth"][lead, c]) for r in recs])
            rc = np.mean([rmse(r["clim"][lead, c], r["truth"][lead, c]) for r in recs])
            ref = min(rp, rc)
            ch_s.append(1.0 - rm / (ref + 1e-12))
            if lead == 3:
                beat_persist_d4.append(rm < rp)
        S = float(np.mean(ch_s))
        pts = w * clip01((S + 0.10) / 0.40)
        out[lead] = dict(skill=S, pts=pts, channel_skills=ch_s)
        total += pts
        print(f"  error day{lead+1}: pooled_skill={S:.3f} pts={pts:.2f}/{w}")
    return total, out, float(np.mean(beat_persist_d4))


def rain_points(recs):
    rain_w = {0: 4.0, 3: 6.0, 7: 5.0}
    total = 0.0
    detail = {}
    for lead, w in rain_w.items():
        csi_m = [csi(r["pred"][lead, RAIN_CH], r["truth"][lead, RAIN_CH], RAIN_THR) for r in recs]
        csi_p = [csi(r["persist"][lead, RAIN_CH], r["truth"][lead, RAIN_CH], RAIN_THR) for r in recs]
        csi_c = [csi(r["clim"][lead, RAIN_CH], r["truth"][lead, RAIN_CH], RAIN_THR) for r in recs]
        mm, pp, cc = float(np.mean(csi_m)), float(np.mean(csi_p)), float(np.mean(csi_c))
        ref = max(pp, cc, 1e-6)
        base = clip01(mm / 0.35) * (w * 0.7)
        beat = clip01((mm - ref) / 0.15 + 0.5) * (w * 0.3)
        pts = min(w, base + beat)
        total += pts
        detail[lead] = dict(model=mm, persist=pp, clim=cc, pts=pts)
        print(f"  rain CSI day{lead+1}: model={mm:.3f} persist={pp:.3f} clim={cc:.3f} pts={pts:.2f}/{w}")
    return total, detail


def blur_points(recs):
    blur_scores = []
    for r in recs:
        for ch in (0, 1):
            std_p8 = spatial_std(r["pred"][7, ch])
            std_t8 = spatial_std(r["truth"][7, ch]) + 1e-6
            std_p1 = spatial_std(r["pred"][0, ch]) + 1e-6
            hf_p = highfreq_energy(r["pred"][7, ch])
            hf_t = highfreq_energy(r["truth"][7, ch]) + 1e-12
            blur_scores.append(dict(
                ratio_truth=std_p8 / std_t8,
                collapse=std_p8 / std_p1,
                hf_ratio=hf_p / hf_t,
            ))
    rt = float(np.mean([s["ratio_truth"] for s in blur_scores]))
    col = float(np.mean([s["collapse"] for s in blur_scores]))
    hf = float(np.mean([s["hf_ratio"] for s in blur_scores]))
    pts = (
        5.0 * clip01(1.0 - abs(rt - 1.0) / 0.7)
        + 3.0 * clip01((col - 0.35) / 0.5)
        + 2.0 * clip01((hf - 0.25) / 0.6)
    )
    print(f"BLUR std_vs_truth={rt:.3f} std_d8/d1={col:.3f} hf_vs_truth={hf:.3f} pts={pts:.2f}/10")
    return pts, dict(std_vs_truth=rt, std_d8_over_d1=col, hf_vs_truth=hf)


def shock_locality(delta, mask, chs):
    d1 = np.sum(np.abs(delta[0, chs]), axis=0)
    d8 = np.sum(np.abs(delta[7, chs]), axis=0)
    m1 = float(d1[mask].sum() / (d1.sum() + 1e-12))
    m8 = float(d8[mask].sum() / (d8.sum() + 1e-12))
    inside = float(d1[mask].mean())
    outside = float(d1[~mask].mean() + 1e-12)
    mag1 = float(d1.mean())
    mag8 = float(d8.mean())
    stable = bool(np.isfinite(delta).all() and mag8 < mag1 * 20 + 1e-6 and mag8 > mag1 * 0.05)
    return dict(mass1=m1, mass8=m8, contrast=inside / outside, spread=m1 - m8,
                mag1=mag1, mag8=mag8, stable=stable)


def shock_pts(s):
    loc = 2.5 * clip01((s["contrast"] - 1.2) / 4.0) + 1.0 * clip01((s["mass1"] - 0.25) / 0.5)
    spr = 1.5 * clip01(s["spread"] / 0.15)
    stb = 1.0 if s["stable"] else 0.0
    reacted = 1.0 if s["mag1"] > 1e-4 else 0.0
    return loc + spr + stb + reacted


def find_ckpt() -> Path:
    for p in CKPT_CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError("monsoon_net_best.pt not found in Downloads or artifacts/")


def main():
    print("=== Independent eval of monsoon_net_best.pt ===")
    ckpt_path = find_ckpt()
    print(f"weights: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError(f"Unexpected checkpoint type/keys: {type(ckpt)} {list(ckpt)[:12] if isinstance(ckpt, dict) else ''}")

    base_ch = int(ckpt.get("base_ch", 48))
    model = MonsoonNet(c_in=int(ckpt.get("c_in", C_IN)), base_ch=base_ch)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        print("MISSING keys", missing)
    if unexpected:
        print("UNEXPECTED keys (ignored)", list(unexpected)[:12], "count", len(unexpected))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params} base_ch={base_ch} epochs={ckpt.get('epochs')} phase_h={ckpt.get('phase_h')}")
    print(f"best_metrics={ckpt.get('best_metrics')}")
    print(f"res_gain={model.res_gain.detach().numpy()}")

    reader = DatasetReader()
    dates = reader._dates
    data = reader.data
    geo = build_geo_features(reader.static[0], reader.static[1], reader.lats, reader.lons)
    tf = Transform()

    train_mask = (dates >= np.datetime64("2005-01-01")) & (dates <= np.datetime64("2017-12-31"))
    clim: dict[int, list] = {}
    for i in np.where(train_mask)[0]:
        d = doy_of(dates, int(i))
        clim.setdefault(d, []).append(data[i])
    clim_mean = {d: np.mean(np.stack(v, 0), axis=0) for d, v in clim.items()}

    recs = []
    t_infer = []
    valid_ok = True
    validity_notes = []

    def roll(init, doy, pert=None):
        return monsoon_rollout_phys(model, tf, geo, init, sincos(doy), 10, pert)

    for etype, day_list in EVENTS.items():
        for ds in day_list:
            t = date_to_idx(dates, ds)
            init = data[t - 2 : t + 1].copy()
            truth = data[t + 1 : t + 11].copy()
            doy = doy_of(dates, t)
            t0 = time.perf_counter()
            pred = roll(init, doy, None)
            t_infer.append((time.perf_counter() - t0) * 1000)
            persist = np.repeat(init[-1][None], 10, axis=0)
            clim_roll = np.stack([
                clim_mean.get(doy_of(dates, t + 1 + k), clim_mean[min(365, doy_of(dates, t + 1 + k))])
                for k in range(10)
            ])
            if not np.isfinite(pred).all():
                valid_ok = False
                validity_notes.append(f"nonfinite {etype} {ds}")
            if pred[:, RAIN_CH].min() < -1e-6:
                valid_ok = False
                validity_notes.append(f"neg rain {etype} {ds}")
            recs.append(dict(etype=etype, date=ds, pred=pred, truth=truth, persist=persist, clim=clim_roll, init=init))

    print(f"windows={len(recs)} mean_infer_ms={np.mean(t_infer):.1f} max_ms={np.max(t_infer):.1f} valid={valid_ok}")

    error_pts, lead_pts, d4_beat = error_points_from_mean_rmse(recs)
    print(f"ERROR {error_pts:.2f}/30  day4_beat_persist={d4_beat:.3f}")
    print("  physical RMSE (model / persist / clim):")
    for c, name in enumerate(CHANNELS):
        row = []
        for lead in LEADS:
            rm = np.mean([rmse(r["pred"][lead, c], r["truth"][lead, c]) for r in recs])
            rp = np.mean([rmse(r["persist"][lead, c], r["truth"][lead, c]) for r in recs])
            rc = np.mean([rmse(r["clim"][lead, c], r["truth"][lead, c]) for r in recs])
            row.append(f"d{lead+1} {rm:.3f}/{rp:.3f}/{rc:.3f}")
        print(f"    {name:5s} " + " | ".join(row))

    rain_pts, rain_detail = rain_points(recs)
    print(f"HEAVY RAIN {rain_pts:.2f}/15")
    forecast_pts = error_pts + rain_pts
    print(f"FORECAST {forecast_pts:.2f}/45")

    blur_pts, blur_diag = blur_points(recs)

    shock_win = next(r for r in recs if r["date"] == "2020-07-15")
    init = shock_win["init"]
    doy = doy_of(dates, date_to_idx(dates, "2020-07-15"))
    latg = reader.lats[:, None]
    long = reader.lons[None, :]
    bob = ((latg >= 8.0) & (latg <= 20.0) & (long >= 85.0) & (long <= 95.0))
    asw = ((latg >= 8.0) & (latg <= 22.0) & (long >= 65.5) & (long <= 75.0))
    zero = np.zeros((6, 32, 36), np.float32)
    warm = np.zeros((6, 32, 36), np.float32)
    warm[0][bob] = 2.0
    warm[5][bob] = 4.0
    weak = np.zeros((6, 32, 36), np.float32)
    weak[2][asw] = -2.0

    t0 = time.perf_counter()
    base = roll(init, doy, None)
    zrun = roll(init, doy, zero)
    wrun = roll(init, doy, warm)
    urun = roll(init, doy, weak)
    print(f"shock rollouts ms={(time.perf_counter() - t0) * 1000:.0f}")

    zdiff = float(np.max(np.abs(zrun - base)))
    zero_pts = 5.0 if zdiff < 1e-5 else (3.0 if zdiff < 1e-3 else (1.0 if zdiff < 0.05 else 0.0))
    print(f"  zero-shock max|d|={zdiff:.3e} pts={zero_pts:.1f}/5")
    sw = shock_locality(wrun - base, bob, [0, 5])
    su = shock_locality(urun - base, asw, [2])
    print("  warm BoB", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sw.items()})
    print("  weak AS ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in su.items()})
    causal = min(10.0, 0.5 * (shock_pts(sw) + shock_pts(su)))
    shock_total = zero_pts + causal
    print(f"  causal combined {causal:.2f}/10")
    print(f"SHOCKS {shock_total:.2f}/15")

    dyn_pts = blur_pts + shock_total
    total = forecast_pts + dyn_pts
    if not valid_ok:
        print("VALIDITY FAIL", validity_notes)
        total *= 0.5

    out = {
        "model": "monsoon_net_best.pt",
        "ckpt_path": str(ckpt_path),
        "n_params": n_params,
        "n_windows": len(recs),
        "mean_infer_ms": float(np.mean(t_infer)),
        "valid": valid_ok,
        "forecast": {"error": error_pts, "rain": rain_pts, "total": forecast_pts, "d4_beat_persist_frac": d4_beat},
        "dynamics": {"blur": blur_pts, "shocks": shock_total, "zero_maxabs": zdiff, "bob": sw, "asw": su, "total": dyn_pts},
        "score_70": total,
        "blur_diagnostics": blur_diag,
        "lead_skills": {str(k): v["skill"] for k, v in lead_pts.items()},
        "rain_detail": {str(k): v for k, v in rain_detail.items()},
        "grast_published_score_70": 41.0,
        "rubric_note": "Same 10 windows, climatology, shocks, CSI, blur, and calibrated error map as GRAST 41/70.",
    }
    print("\n==== SCORE ====")
    print(json.dumps({k: out[k] for k in ["forecast", "dynamics", "score_70", "mean_infer_ms", "valid"]}, indent=2, default=float))
    out_path = ROOT / "artifacts" / "eval_monsoon_net_independent.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
