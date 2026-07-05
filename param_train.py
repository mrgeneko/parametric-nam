#!/usr/bin/env python3
"""
param_train.py — Train a parametric A2 + FiLM Neural Amp Modeler.

Loads a dataset produced by batch_generate.py, trains an A2 architecture
with FiLM conditioning on knob parameters, and exports a .param.nam file
loadable by the C++ ParametricWaveNet subclass.

Usage:
  python param_train.py --dataset /path/to/dataset --output model.param.nam
"""

import argparse, csv, datetime, json, math, os, sys, time, warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# A2 architecture constants  (must match C++ a2_fast.h exactly)
# ---------------------------------------------------------------------------
K_NUM_LAYERS = 23
K_HEAD_KERNEL = 16
K_LEAKY_SLOPE = 0.01

K_KERNEL_SIZES = [
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    15, 15,
    6, 6, 6, 6, 6, 6, 6,
]

K_DILATIONS = [
    1, 3, 7, 17, 41, 101, 239,
    1, 3, 7, 17, 41, 101, 239,
    1, 13,
    1, 3, 7, 17, 41, 101, 239,
]

# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class FiLM(nn.Module):
    """Feature-wise Linear Modulation: gamma * x + beta."""
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.net = nn.Linear(cond_dim, 2 * channels)
        # Identity-like init: gamma≈1, beta≈0 so FiLM passes x unchanged at start
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)
        self.net.bias.data[:channels].fill_(1.0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        params = self.net(cond)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return gamma * x + beta


class A2Layer(nn.Module):
    """Single A2 dilated conv layer with optional FiLM."""
    def __init__(self, channels: int, kernel_size: int, dilation: int, cond_dim: int):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              padding=pad, dilation=dilation)
        self.mixin = nn.Conv1d(1, channels, 1, bias=False)
        self.film = FiLM(channels, cond_dim) if cond_dim > 0 else None
        self.l1x1 = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor, inp_audio: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.conv(x)
        h = h[:, :, :inp_audio.shape[-1]]
        h = h + self.mixin(inp_audio)
        if self.film is not None:
            h = self.film(h, cond)
        h = F.leaky_relu(h, K_LEAKY_SLOPE)
        h = self.l1x1(h)
        return h + residual


class ParametricA2(nn.Module):
    """A2 (LeakyReLU, mixed kernels) with FiLM conditioning on knob params.

    Architecture matches the C++ A2FastModel with additional FiLM layers
    for parametric knob control.
    """
    def __init__(self, channels: int, num_params: int):
        super().__init__()
        self.channels = channels
        self.num_params = num_params
        self.rechannel = nn.Conv1d(1, channels, 1, bias=False)
        self.layers = nn.ModuleList([
            A2Layer(channels, K_KERNEL_SIZES[i], K_DILATIONS[i], num_params)
            for i in range(K_NUM_LAYERS)
        ])
        self.head = nn.Conv1d(channels, 1, K_HEAD_KERNEL, padding=K_HEAD_KERNEL // 2)
        self.head_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, audio: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        x = self.rechannel(audio)
        inp_audio = audio
        for layer in self.layers:
            x = layer(x, inp_audio, params)
        x = self.head(x)
        x = x[:, :, :audio.shape[-1]]
        x = x * self.head_scale
        return x

    def weight_count(self) -> int:
        """Total flat float count for .nam export matching C++ order."""
        total = 0
        total += self.rechannel.weight.numel()
        for layer in self.layers:
            total += layer.conv.weight.numel()
            total += layer.conv.bias.numel()
            total += layer.mixin.weight.numel()
            total += layer.l1x1.weight.numel()
            total += layer.l1x1.bias.numel()
            if layer.film is not None:
                total += layer.film.net.weight.numel()
                total += layer.film.net.bias.numel()
        total += self.head.weight.numel()
        total += self.head.bias.numel()
        total += 1  # head_scale
        return total

    def _export_weight_block(self) -> list[float]:
        """C++ weight order — Conv layers: out_ch→in_ch→tap (r-major)."""
        w = []
        w.extend(self.rechannel.weight.detach().flatten().tolist())
        for layer in self.layers:
            w.extend(layer.conv.weight.detach().flatten().tolist())
            w.extend(layer.conv.bias.detach().flatten().tolist())
            w.extend(layer.mixin.weight.detach().flatten().tolist())
            w.extend(layer.l1x1.weight.detach().flatten().tolist())
            w.extend(layer.l1x1.bias.detach().flatten().tolist())
            if layer.film is not None:
                w.extend(layer.film.net.weight.detach().flatten().tolist())
                w.extend(layer.film.net.bias.detach().flatten().tolist())
        w.extend(self.head.weight.detach().flatten().tolist())
        w.extend(self.head.bias.detach().flatten().tolist())
        w.append(self.head_scale.detach().item())
        return w

    def _load_weight_block(self, it):
        """Consume flat iterator in C++ order, populate weights in-place."""
        self.rechannel.weight.data.copy_(
            torch.tensor([next(it) for _ in range(self.channels)]).view_as(self.rechannel.weight))
        for layer in self.layers:
            C = self.channels
            K = layer.conv.weight.shape[-1]
            layer.conv.weight.data.copy_(
                torch.tensor([next(it) for _ in range(K * C * C)]).view_as(layer.conv.weight))
            layer.conv.bias.data.copy_(
                torch.tensor([next(it) for _ in range(C)]).view_as(layer.conv.bias))
            layer.mixin.weight.data.copy_(
                torch.tensor([next(it) for _ in range(C)]).view_as(layer.mixin.weight))
            layer.l1x1.weight.data.copy_(
                torch.tensor([next(it) for _ in range(C * C)]).view_as(layer.l1x1.weight))
            layer.l1x1.bias.data.copy_(
                torch.tensor([next(it) for _ in range(C)]).view_as(layer.l1x1.bias))
            if layer.film is not None:
                np_ = self.num_params
                layer.film.net.weight.data.copy_(
                    torch.tensor([next(it) for _ in range(2 * C * np_)]).view_as(layer.film.net.weight))
                layer.film.net.bias.data.copy_(
                    torch.tensor([next(it) for _ in range(2 * C)]).view_as(layer.film.net.bias))
        self.head.weight.data.copy_(
            torch.tensor([next(it) for _ in range(K_HEAD_KERNEL * self.channels)]).view_as(self.head.weight))
        self.head.bias.data.copy_(
            torch.tensor([next(it) for _ in range(1)]).view_as(self.head.bias))
        self.head_scale.data.copy_(torch.tensor(next(it)))

    def export_weights(self) -> list[float]:
        return self._export_weight_block()

    def load_weights(self, weights: list[float]):
        self._load_weight_block(iter(weights))

    def export_nam(self, config: dict, metadata: dict, sample_rate: int,
                   input_audio: "np.ndarray | None" = None) -> dict:
        weights = self.export_weights()

        # Compute gain: run at default knobs (all 0.5), target -18 dBFS output.
        # gain is the scalar the plugin multiplies output by for level normalisation.
        gain = 1.0
        if input_audio is not None:
            device = next(self.parameters()).device
            clip = input_audio[:sample_rate * 10]  # 10 s is enough for RMS
            with torch.no_grad():
                inp = torch.from_numpy(clip).float().unsqueeze(0).unsqueeze(0).to(device)
                params = torch.full((1, self.num_params), 0.5, device=device)
                out = self(inp, params)
            out_rms = float(out.squeeze().cpu().pow(2).mean().sqrt().item())
            target_rms = 10 ** (-18.0 / 20.0)
            if out_rms > 1e-8:
                gain = float(target_rms / out_rms)

        # Per-knob defs. A knob listed in config["steps"] is a discrete N-position
        # switch: emit a "steps" hint so the host renders a stepped control and
        # quantizes the value to N positions {0, 1/(N-1), ..., 1}. Absent = continuous.
        steps_map = config.get("steps", {}) or {}
        def _default_for(name):
            n = steps_map.get(name)
            if not n:
                return 0.5
            positions = [i / (n - 1) for i in range(n)]
            return min(positions, key=lambda p: abs(p - 0.5))  # nearest step to 0.5
        param_defs = []
        for name in config.get("param_names", []):
            d = {"name": name, "min": 0.0, "max": 1.0, "default": _default_for(name)}
            if name in steps_map:
                d["steps"] = int(steps_map[name])
            param_defs.append(d)
        layer_config = {
            "layers": self.channels,
            "head_scale": self.head_scale.item(),
            "parametric": {
                "type": "film",
                "condition_size": self.num_params,
                "film_layers": [f"layer_{i}" for i in range(K_NUM_LAYERS)],
                "parameters": param_defs,
            },
        }
        now = datetime.datetime.now()
        nam_metadata = {
            "date": {
                "year": now.year, "month": now.month, "day": now.day,
                "hour": now.hour, "minute": now.minute, "second": now.second,
            },
            "loudness": -18.0,
            "gain": gain,
            "name": config.get("name", config.get("circuit", "")),
            "modeled_by": config.get("modeled_by", ""),
            "gear_make": config.get("gear_make", ""),
            "gear_model": config.get("gear_model", config.get("circuit", "")),
            "gear_type": config.get("gear_type", "amp"),
            "tone_type": config.get("tone_type", ""),
        }
        # Strip empty strings so absent fields don't clutter the file
        nam_metadata = {k: v for k, v in nam_metadata.items() if v != ""}
        return {
            "version": metadata.get("version", "0.7.0"),
            "architecture": "ParametricWaveNet",
            "config": layer_config,
            "weights": weights,
            "metadata": nam_metadata,
            "sample_rate": sample_rate,
        }


# ---------------------------------------------------------------------------
# Slimmable model (A2 Lite 3ch + A2 Full 8ch, trained jointly)
# ---------------------------------------------------------------------------

K_LITE_CHANNELS = 3
K_FULL_CHANNELS = 8


class SlimmableParametricA2(nn.Module):
    """Two independent ParametricA2 models trained simultaneously.

    Exports as SlimmableParametricContainer with:
      max_value=0.5 → lite (3ch)
      max_value=1.0 → full (8ch)
    """
    def __init__(self, num_params: int):
        super().__init__()
        self.lite = ParametricA2(K_LITE_CHANNELS, num_params)
        self.full = ParametricA2(K_FULL_CHANNELS, num_params)
        self.num_params = num_params

    def forward(self, audio: torch.Tensor, params: torch.Tensor):
        return self.lite(audio, params), self.full(audio, params)

    def export_nam(self, config: dict, metadata: dict, sample_rate: int,
                   input_audio: "np.ndarray | None" = None) -> dict:
        lite_nam = self.lite.export_nam(config, metadata, sample_rate, input_audio)
        full_nam = self.full.export_nam(config, metadata, sample_rate, input_audio)
        return {
            "version": metadata.get("version", "0.7.0"),
            "architecture": "SlimmableContainer",
            "config": {
                "submodels": [
                    {"max_value": 0.5, "model": lite_nam},
                    {"max_value": 1.0, "model": full_nam},
                ]
            },
            "weights": [],
            "metadata": {
                **{k: v for k, v in full_nam["metadata"].items()
                   if k not in ("gain",)},
                "gain": full_nam["metadata"]["gain"],
            },
            "sample_rate": sample_rate,
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ParamDataset(torch.utils.data.Dataset):
    """Loads dataset produced by batch_harness.py (livespice backend).

    Expected directory structure:
        dataset/
            sweep.wav        — input audio used during capture
            outputs.npy      — float32 [N_perms, N_samples] (run --combine first)
            params.csv       — idx, knob1, ..., ok, error columns
            config.json      — must contain a "knobs" list
    """
    def __init__(self, dataset_dir: str, crop_len: int = 48000, repeats: int = 1,
                 target_dbfs: float = -18.0, mmap: bool = False):
        self.dir = Path(dataset_dir)
        with open(self.dir / "config.json") as f:
            self.config = json.load(f)
        self.param_names = self.config["knobs"]
        self.config["param_names"] = self.param_names  # alias for export_nam
        self.num_params = len(self.param_names)
        self.crop_len = crop_len
        self.repeats = repeats

        # Load input sweep and normalize to target_dbfs RMS
        inp_raw, _ = sf.read(str(self.dir / "sweep.wav"))
        if inp_raw.ndim > 1:
            inp_raw = inp_raw.mean(axis=1)
        inp_raw = inp_raw.astype(np.float32)
        rms = float(np.sqrt(np.mean(inp_raw ** 2)))
        target_rms = 10 ** (target_dbfs / 20.0)
        self._scale = target_rms / (rms + 1e-8)
        self.inp = inp_raw * self._scale

        # Load combined outputs. Default: fully into RAM for fastest random
        # access (dataset must fit in RAM). With mmap=True: memory-map from disk
        # so the dataset can exceed RAM — __getitem__ touches one perm row at a
        # time and the OS page-caches hot rows, at some random-access I/O cost.
        out_path = self.dir / "outputs.npy"
        if not out_path.exists():
            raise FileNotFoundError(
                f"outputs.npy not found. Run first:\n"
                f"  python batch_harness.py --combine {self.dir}"
            )
        if mmap:
            print(f"  Memory-mapping outputs.npy (--mmap; disk-bound, not RAM) ...",
                  file=sys.stderr, flush=True)
            self.outputs = np.load(str(out_path), mmap_mode="r")
        else:
            print(f"  Loading outputs.npy into RAM ...", file=sys.stderr, flush=True)
            self.outputs = np.load(str(out_path))

        # Load params.csv — successful rows only, ordered by permutation idx
        rows = []
        with open(self.dir / "params.csv", newline="") as f:
            for row in csv.DictReader(f):
                if int(row["ok"]) == 1:
                    rows.append(row)
        rows.sort(key=lambda r: int(r["idx"]))
        self.samples = [
            (int(r["idx"]), {n: float(r[n]) for n in self.param_names})
            for r in rows
        ]

        self._check_output_diversity()

    def _check_output_diversity(self) -> None:
        """Spot-check a sample of output pairs for accidental duplicates."""
        if len(self.samples) < 2:
            return
        rng = np.random.default_rng(42)
        check = rng.choice(len(self.samples), size=min(20, len(self.samples)), replace=False)
        for i in range(len(check)):
            for j in range(i + 1, len(check)):
                idx_i = self.samples[int(check[i])][0]
                idx_j = self.samples[int(check[j])][0]
                if np.abs(self.outputs[idx_i] - self.outputs[idx_j]).max() < 1e-7:
                    warnings.warn(
                        f"Outputs {idx_i} and {idx_j} are identical "
                        f"— dataset may be invalid. Check param_map names "
                        "in batch_harness.py match circuit component names."
                    )

    def __len__(self):
        return len(self.samples) * self.repeats

    def __getitem__(self, idx):
        real_idx = idx % len(self.samples)
        perm_idx, params = self.samples[real_idx]

        inp = self.inp
        out = (self.outputs[perm_idx] * self._scale).astype(np.float32)

        sig_len = min(len(inp), len(out))
        if sig_len > self.crop_len:
            start = np.random.randint(0, sig_len - self.crop_len)
            inp = inp[start:start + self.crop_len]
            out = out[start:start + self.crop_len]
        else:
            pad = self.crop_len - sig_len
            inp = np.pad(inp[:sig_len], (0, pad))
            out = np.pad(out[:sig_len], (0, pad))

        inp_t = torch.from_numpy(inp.copy()).float().unsqueeze(0)
        out_t = torch.from_numpy(out.copy()).float().unsqueeze(0)
        params_t = torch.tensor([params[n] for n in self.param_names]).float()
        return inp_t, out_t, params_t


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def esr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Error-to-signal ratio."""
    return ((pred - target) ** 2).sum() / (target ** 2).sum()


def mrstft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Multi-resolution STFT loss (L1 on magnitudes)."""
    fft_sizes = [512, 1024, 2048]
    hop_sizes = [128, 256, 512]
    win_lengths = [512, 1024, 2048]
    total = torch.tensor(0.0, device=pred.device)
    for fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths):
        s_pred = torch.stft(pred.squeeze(1), fft, hop, win,
                            torch.hann_window(win, device=pred.device),
                            return_complex=True)
        s_target = torch.stft(target.squeeze(1), fft, hop, win,
                              torch.hann_window(win, device=pred.device),
                              return_complex=True)
        mag_pred = torch.abs(s_pred)
        mag_target = torch.abs(s_target)
        total += F.l1_loss(mag_pred, mag_target)
    return total / len(fft_sizes)


class ParamLoss(nn.Module):
    """Combined MSE + MRSTFT loss."""
    def __init__(self, mrstft_weight: float = 0.1):
        super().__init__()
        self.mrstft_weight = mrstft_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        mrstft = mrstft_loss(pred, target)
        return mse + self.mrstft_weight * mrstft


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, clip_norm=1.0,
                epoch: int = 0, total_epochs: int = 0, log_interval: int = 50):
    model.train()
    slimmable = isinstance(model, SlimmableParametricA2)
    total_loss = 0
    for step, (inp, out, params) in enumerate(loader):
        inp, out, params = inp.to(device), out.to(device), params.to(device)
        optimizer.zero_grad()
        if slimmable:
            pred_lite, pred_full = model(inp, params)
            loss = criterion(pred_lite, out) + criterion(pred_full, out)
        else:
            pred = model(inp, params)
            loss = criterion(pred, out)
        loss.backward()
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total_loss += loss.item()
        if log_interval > 0 and (step + 1) % log_interval == 0:
            print(f"  [{epoch:3d}/{total_epochs}] step {step+1}/{len(loader)}  "
                  f"loss={total_loss/(step+1):.6f}", file=sys.stderr, flush=True)
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    slimmable = isinstance(model, SlimmableParametricA2)
    total_loss = 0
    total_esr = 0
    n = 0
    with torch.no_grad():
        for inp, out, params in loader:
            inp, out, params = inp.to(device), out.to(device), params.to(device)
            if slimmable:
                pred_lite, pred_full = model(inp, params)
                total_loss += (criterion(pred_lite, out) + criterion(pred_full, out)).item()
                # Report ESR of the full model as the primary metric
                total_esr += esr(pred_full, out).item()
            else:
                pred = model(inp, params)
                total_loss += criterion(pred, out).item()
                total_esr += esr(pred, out).item()
            n += 1
    return total_loss / n, total_esr / n


# ---------------------------------------------------------------------------
# Parameter sensitivity check
# ---------------------------------------------------------------------------

def param_sensitivity(model: ParametricA2, device: str,
                      sweep: np.ndarray, n_steps: int = 11) -> dict:
    """Sweep each knob from 0→1, measure output change.

    Returns dict of {param_name: max_output_diff}.
    If any diff is below threshold, the model may not have learned
    parametric behavior.
    """
    model.eval()
    inp = torch.from_numpy(sweep).float().unsqueeze(0).unsqueeze(0).to(device)
    results: dict[str, float] = {}
    with torch.no_grad():
        ref_params = torch.full((1, model.num_params), 0.5, device=device)
        ref_out = model(inp, ref_params)
        for pi in range(model.num_params):
            max_diff = 0.0
            for step in range(n_steps):
                p = step / (n_steps - 1)
                params = torch.full((1, model.num_params), 0.5, device=device)
                params[0, pi] = p
                out = model(inp, params)
                diff = (out - ref_out).abs().max().item()
                max_diff = max(max_diff, diff)
            results[f"param_{pi}"] = max_diff
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Train parametric A2 + FiLM NAM model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset", required=True, type=Path,
                    help="Dataset directory from batch_generate.py")
    ap.add_argument("--output", "-o", required=True, type=Path,
                    help="Output .param.nam file path")
    ap.add_argument("--channels", type=int, default=8,
                    help="A2 channel count (3=nano, 8=standard) (default: %(default)s)")
    ap.add_argument("--epochs", type=int, default=100,
                    help="Number of training epochs (default: %(default)s)")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Batch size (default: %(default)s)")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="Learning rate (default: %(default)s)")
    ap.add_argument("--crop-len", type=int, default=44100,
                    help="Random crop length in samples (default: %(default)s)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Virtual dataset multiplier — increases steps/epoch without "
                         "changing the audio data (default: %(default)s)")
    ap.add_argument("--mmap", action="store_true",
                    help="Memory-map outputs.npy from disk instead of loading it fully "
                         "into RAM. Use for datasets larger than RAM (disk-bound; slower "
                         "random access but the OS page-caches hot rows).")
    ap.add_argument("--val-split", type=float, default=0.1,
                    help="Fraction of samples for validation (default: %(default)s)")
    ap.add_argument("--mrstft-weight", type=float, default=0.1,
                    help="MRSTFT loss weight (default: %(default)s)")
    ap.add_argument("--device", default="auto",
                    help="Device: auto, cpu, cuda, mps (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: %(default)s)")
    ap.add_argument("--slimmable", action="store_true",
                    help="Train A2 Lite (3ch) + A2 Full (8ch) jointly and export as "
                         "SlimmableParametricContainer (ignores --channels)")
    ap.add_argument("--param-sensitivity", action="store_true",
                    help="Run parameter sensitivity check after training")
    ap.add_argument("--checkpoint-dir", type=Path, default=None,
                    help="Directory to save epoch checkpoints and metrics CSV")
    ap.add_argument("--resume", type=Path, default=None,
                    help="Checkpoint .pt to resume from")
    ap.add_argument("--log-csv", type=Path, default=None,
                    help="Path for metrics CSV (default: --checkpoint-dir/metrics.csv)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Device: {device}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print(f"\nLoading dataset from {args.dataset} ...", file=sys.stderr)
    dataset = ParamDataset(str(args.dataset), crop_len=args.crop_len, repeats=args.repeats,
                           mmap=args.mmap)
    n_total = len(dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"  {n_total} total samples ({n_train} train, {n_val} val)",
          file=sys.stderr)
    print(f"  Params: {dataset.param_names}", file=sys.stderr)

    ckpt_dir = args.checkpoint_dir
    log_csv = args.log_csv
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if log_csv is None:
            log_csv = ckpt_dir / "metrics.csv"

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    num_params = dataset.num_params
    if args.slimmable:
        model = SlimmableParametricA2(num_params)
        lite_w = model.lite.weight_count()
        full_w = model.full.weight_count()
        print(f"\nModel: SlimmableParametricA2  lite={K_LITE_CHANNELS}ch ({lite_w}w) + "
              f"full={K_FULL_CHANNELS}ch ({full_w}w), {num_params} params", file=sys.stderr)
    else:
        model = ParametricA2(args.channels, num_params)
        n_weights = model.weight_count()
        print(f"\nModel: A2 {args.channels}-channel, {num_params} params, "
              f"{n_weights} weights", file=sys.stderr)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    start_epoch = 1
    best_esr = float("inf")
    best_state = None

    if args.resume is not None:
        print(f"Resuming from {args.resume} ...", file=sys.stderr)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_esr = ckpt.get("best_esr", float("inf"))
        best_state = ckpt.get("best_state", None)
        print(f"  Resumed at epoch {ckpt['epoch']}, best ESR {best_esr:.6f}", file=sys.stderr)

    scheduler_last = start_epoch - 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs,
                                                           last_epoch=scheduler_last)
    if args.resume and "scheduler_last_epoch" in ckpt:
        scheduler_last = ckpt["scheduler_last_epoch"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs,
                                                               last_epoch=scheduler_last)
    criterion = ParamLoss(mrstft_weight=args.mrstft_weight)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"\nTraining {args.epochs} epochs ...", file=sys.stderr)
    if log_csv is not None:
        log_f = open(log_csv, "a" if args.resume else "w", newline="")
        log_w = csv.writer(log_f)
        if not args.resume:
            log_w.writerow(["epoch", "train_loss", "val_loss", "val_esr", "lr", "elapsed_s"])

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                 epoch=epoch, total_epochs=args.epochs, log_interval=10)
        val_loss, val_esr = validate(model, val_loader, criterion, device)
        scheduler.step()

        new_best = val_esr < best_esr
        if new_best:
            best_esr = val_esr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if ckpt_dir is not None:
                torch.save({
                    "epoch": epoch,
                    "model": best_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler_last_epoch": scheduler.last_epoch,
                    "best_esr": best_esr,
                    "args_dict": dict(vars(args)),
                }, ckpt_dir / "best.pt")
            # Export best .param.nam — swap in best weights, export, restore current weights
            current_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            model.to(device)
            nam_data = model.export_nam(dataset.config, {"version": "0.7.0"}, sample_rate=48000,
                                        input_audio=dataset.inp)
            nam_path = args.output.parent / (args.output.stem + "_best" + args.output.suffix)
            nam_path.write_text(json.dumps(nam_data, separators=(",", ":")))
            model.load_state_dict(current_state)
            model.to(device)

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        # Print every epoch for production monitoring
        eta = (elapsed / (epoch - start_epoch + 1)) * (args.epochs - epoch)
        best_marker = " *" if new_best else ""
        print(f"  [{epoch:3d}/{args.epochs}]  "
              f"train={train_loss:.6f}  val_loss={val_loss:.6f}  "
              f"val_ESR={val_esr:.6f}  lr={lr_now:.2e}  "
              f"({elapsed:.0f}s, ETA {eta:.0f}s){best_marker}",
              file=sys.stderr, flush=True)

        # Log CSV
        if log_csv is not None:
            log_w.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}",
                            f"{val_esr:.8f}", f"{lr_now:.2e}", f"{elapsed:.1f}"])
            log_f.flush()

        # Save checkpoint every epoch (overwrite previous to save disk space)
        if ckpt_dir is not None:
            ckpt_path = ckpt_dir / "latest.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler_last_epoch": scheduler.last_epoch,
                "best_esr": best_esr,
                "best_state": best_state,
                "args_dict": dict(vars(args)),
            }, ckpt_path)

    elapsed = time.time() - t0
    if log_csv is not None:
        log_f.close()

    print(f"\nTraining finished ({elapsed:.0f}s, {elapsed/args.epochs:.1f}s/epoch)",
          file=sys.stderr)
    print(f"Best validation ESR: {best_esr:.6f}", file=sys.stderr)

    # Restore best state
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    # Save best checkpoint
    if ckpt_dir is not None:
        best_path = ckpt_dir / "best.pt"
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler_last_epoch": scheduler.last_epoch,
            "best_esr": best_esr,
            "args_dict": dict(vars(args)),
        }, best_path)
        print(f"  Best model saved to {best_path}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Parameter sensitivity check
    # ------------------------------------------------------------------
    if args.param_sensitivity:
        print(f"\nParameter sensitivity check ...", file=sys.stderr)
        sweep_audio, _ = sf.read(str(dataset.dir / "sweep.wav"))
        if sweep_audio.ndim > 1:
            sweep_audio = sweep_audio.mean(axis=1)
        sweep_audio = sweep_audio[:48000]
        targets = [model.full] if args.slimmable else [model]
        labels = ["full"] if args.slimmable else [""]
        if args.slimmable:
            targets.append(model.lite)
            labels.append("lite")
        for m, lbl in zip(targets, labels):
            prefix = f"[{lbl}] " if lbl else ""
            sens = param_sensitivity(m, device, sweep_audio)
            for k, v in sens.items():
                pname = dataset.param_names[int(k.split("_")[-1])] if "_" in k else k
                print(f"  {prefix}{pname}: max_diff = {v:.6f}", file=sys.stderr)
                if v < 1e-6:
                    warnings.warn(f"  {prefix}{pname}: output doesn't change with this param "
                                  f"(max_diff={v:.2e}) — model may be ignoring knobs")

    # ------------------------------------------------------------------
    # Export .param.nam
    # ------------------------------------------------------------------
    print(f"\nExporting to {args.output} ...", file=sys.stderr)
    nam_data = model.export_nam(dataset.config, {"version": "0.7.0"}, sample_rate=48000,
                                input_audio=dataset.inp)
    args.output.write_text(json.dumps(nam_data, separators=(",", ":")))
    print(f"  Architecture: {nam_data['architecture']}", file=sys.stderr)
    if args.slimmable:
        for sm in nam_data["config"]["submodels"]:
            w = len(sm["model"]["weights"])
            ch = sm["model"]["config"]["layers"]
            print(f"    max_value={sm['max_value']}  {ch}ch  {w} weights", file=sys.stderr)
    else:
        print(f"  Exported {len(nam_data['weights'])} weights", file=sys.stderr)

    # Verify round-trip
    print(f"  Verifying round-trip ...", file=sys.stderr)
    test_inp = torch.randn(1, 1, 8192, device=device)
    test_params = torch.rand(1, num_params, device=device)
    if args.slimmable:
        for sm_data, lbl in zip(nam_data["config"]["submodels"], ["lite", "full"]):
            ch = sm_data["model"]["config"]["layers"]
            m2 = ParametricA2(ch, num_params)
            m2.load_weights(sm_data["model"]["weights"])
            m2.to(device)
            src = model.lite if lbl == "lite" else model.full
            with torch.no_grad():
                o1 = src(test_inp, test_params)
                o2 = m2(test_inp, test_params)
            md = (o1 - o2).abs().max().item()
            status = f"OK (max_diff={md:.2e})" if md <= 1e-6 else f"WARN max_diff={md:.2e}"
            print(f"    [{lbl}] round-trip {status}", file=sys.stderr)
    else:
        model2 = ParametricA2(args.channels, num_params)
        model2.load_weights(nam_data["weights"])
        model2.to(device)
        with torch.no_grad():
            o1 = model(test_inp, test_params)
            o2 = model2(test_inp, test_params)
        md = (o1 - o2).abs().max().item()
        if md > 1e-6:
            warnings.warn(f"Round-trip max diff = {md:.2e} — weight export may be corrupt")
        else:
            print(f"  Round-trip OK (max diff = {md:.2e})", file=sys.stderr)

    print(f"\nDone. Model saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
