#!/usr/bin/env python3
"""
param_train.py — Train a parametric A2 + FiLM Neural Amp Modeler.

Loads a dataset produced by batch_generate.py, trains an A2 architecture
with FiLM conditioning on knob parameters, and exports a .param.nam file
loadable by the C++ ParametricWaveNet subclass.

Usage:
  python param_train.py --dataset /path/to/dataset --output model.param.nam
"""

import argparse, json, math, os, sys, time, warnings
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

    def export_nam(self, config: dict, metadata: dict, sample_rate: int) -> dict:
        weights = self.export_weights()
        param_defs = [
            {"name": name, "min": 0.0, "max": 1.0, "default": 0.5}
            for name in config.get("param_names", [])
        ]
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
        return {
            "version": metadata.get("version", "0.7.0"),
            "architecture": "ParametricWaveNet",
            "config": layer_config,
            "weights": weights,
            "metadata": {
                "source_circuit": config.get("circuit", ""),
                "source_schx": config.get("circuit", "") + ".schx",
                "loudness": -18.0,
            },
            "sample_rate": sample_rate,
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ParamDataset(torch.utils.data.Dataset):
    """Loads paired sweep + output dataset from batch_generate.py output.

    Directory structure:
        dataset/
            input_-12dBFS.wav
            input_-6dBFS.wav
            params.csv
            config.json
            samples/
                000000/
                    params.json
                    output_-12dBFS.wav
                    output_-6dBFS.wav
                000001/
                    ...
    """
    def __init__(self, dataset_dir: str, crop_len: int = 48000, repeats: int = 1):
        self.dir = Path(dataset_dir)
        with open(self.dir / "config.json") as f:
            self.config = json.load(f)
        self.param_names = self.config["param_names"]
        self.num_params = len(self.param_names)
        self.levels = self.config["input_levels_dbfs"]
        self.repeats = repeats

        # Load input sweeps
        self.inputs: dict[float, np.ndarray] = {}
        for db in self.levels:
            path = self.dir / f"input_{db:.0f}dBFS.wav"
            data, sr = sf.read(str(path))
            if data.ndim > 1:
                data = data.mean(axis=1)
            self.inputs[db] = data

        # Enumerate all samples and pre-load outputs into memory
        samples_dir = self.dir / "samples"
        self.samples: list[tuple[int, dict]] = []
        self.outputs: dict[tuple[int, float], np.ndarray] = {}
        for d in sorted(samples_dir.iterdir()):
            if not d.is_dir():
                continue
            with open(d / "params.json") as f:
                p = json.load(f)
            sid = int(d.name)
            self.samples.append((sid, p))
            for db in self.levels:
                out, _ = sf.read(str(d / f"output_{db:.0f}dBFS.wav"))
                if out.ndim > 1:
                    out = out.mean(axis=1)
                self.outputs[(sid, db)] = out

        self.crop_len = crop_len
        self._base_len = len(self.samples) * len(self.levels)

    def __len__(self):
        return self._base_len * self.repeats

    def __getitem__(self, idx):
        real_idx = idx % self._base_len
        sidx, level_idx = divmod(real_idx, len(self.levels))
        sample_id, params = self.samples[sidx]
        db = self.levels[level_idx]

        inp = self.inputs[db]
        out = self.outputs[(sample_id, db)]

        # Random crop
        if len(inp) > self.crop_len:
            start = np.random.randint(0, len(inp) - self.crop_len)
            inp = inp[start:start + self.crop_len]
            out = out[start:start + self.crop_len]
        else:
            pad = self.crop_len - len(inp)
            inp = np.pad(inp, (0, pad))
            out = np.pad(out, (0, pad))

        inp_t = torch.from_numpy(inp).float().unsqueeze(0)
        out_t = torch.from_numpy(out).float().unsqueeze(0)
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

def train_epoch(model, loader, optimizer, criterion, device, clip_norm=1.0):
    model.train()
    total_loss = 0
    for inp, out, params in loader:
        inp, out, params = inp.to(device), out.to(device), params.to(device)
        optimizer.zero_grad()
        pred = model(inp, params)
        loss = criterion(pred, out)
        loss.backward()
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_esr = 0
    n = 0
    with torch.no_grad():
        for inp, out, params in loader:
            inp, out, params = inp.to(device), out.to(device), params.to(device)
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
    ap.add_argument("--val-split", type=float, default=0.1,
                    help="Fraction of samples for validation (default: %(default)s)")
    ap.add_argument("--mrstft-weight", type=float, default=0.1,
                    help="MRSTFT loss weight (default: %(default)s)")
    ap.add_argument("--device", default="auto",
                    help="Device: auto, cpu, cuda, mps (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: %(default)s)")
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
    dataset = ParamDataset(str(args.dataset), crop_len=args.crop_len, repeats=args.repeats)
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
    model = ParametricA2(args.channels, num_params)
    model.to(device)
    n_weights = model.weight_count()
    print(f"\nModel: A2 {args.channels}-channel, {num_params} params, "
          f"{n_weights} weights", file=sys.stderr)

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
        import csv as csv_mod
        log_f = open(log_csv, "a" if args.resume else "w", newline="")
        log_w = csv_mod.writer(log_f)
        if not args.resume:
            log_w.writerow(["epoch", "train_loss", "val_loss", "val_esr", "lr", "elapsed_s"])

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_esr = validate(model, val_loader, criterion, device)
        scheduler.step()

        if val_esr < best_esr:
            best_esr = val_esr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        # Print every epoch for production monitoring
        eta = (elapsed / (epoch - start_epoch + 1)) * (args.epochs - epoch) if epoch > start_epoch else 0
        print(f"  [{epoch:3d}/{args.epochs}]  "
              f"train={train_loss:.6f}  val_loss={val_loss:.6f}  "
              f"val_ESR={val_esr:.6f}  lr={lr_now:.2e}  "
              f"({elapsed:.0f}s, ETA {eta:.0f}s)",
              file=sys.stderr, flush=True)

        # Log CSV
        if log_csv is not None:
            log_w.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}",
                            f"{val_esr:.8f}", f"{lr_now:.2e}", f"{elapsed:.1f}"])
            log_f.flush()

        # Save periodic checkpoint
        if ckpt_dir is not None and (epoch == 1 or epoch % 10 == 0 or epoch == args.epochs):
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
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
        sweep, sr = sf.read(str(dataset.dir / "input_-12dBFS.wav"))
        if sweep.ndim > 1:
            sweep = sweep.mean(axis=1)
        sweep = sweep[:48000]
        sens = param_sensitivity(model, device, sweep)
        for k, v in sens.items():
            label = dataset.param_names[int(k.split("_")[-1])] if "_" in k else k
            print(f"  {label}: max_diff = {v:.6f}", file=sys.stderr)
            if v < 1e-6:
                warnings.warn(f"  {label}: output doesn't change with this param "
                              f"(max_diff={v:.2e}) — model may be ignoring knobs")

    # ------------------------------------------------------------------
    # Export .param.nam
    # ------------------------------------------------------------------
    print(f"\nExporting to {args.output} ...", file=sys.stderr)
    nam_data = model.export_nam(dataset.config, {"version": "0.7.0"}, sample_rate=48000)
    args.output.write_text(json.dumps(nam_data, separators=(",", ":")))
    n_exported = len(nam_data["weights"])
    print(f"  Exported {n_exported} weights to {args.output}", file=sys.stderr)

    # Verify round-trip: export → reload in Python → compare forward pass
    print(f"  Verifying round-trip ...", file=sys.stderr)
    model2 = ParametricA2(args.channels, num_params)
    model2.load_weights(nam_data["weights"])
    model2.to(device)
    test_inp = torch.randn(1, 1, 8192, device=device)
    test_params = torch.rand(1, num_params, device=device)
    with torch.no_grad():
        out1 = model(test_inp, test_params)
        out2 = model2(test_inp, test_params)
    max_diff = (out1 - out2).abs().max().item()
    if max_diff > 1e-6:
        warnings.warn(f"Round-trip max diff = {max_diff:.2e} — weight export may be corrupt")
    else:
        print(f"  Round-trip OK (max diff = {max_diff:.2e})", file=sys.stderr)

    print(f"\nDone. Model saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
