"""Entrena desde cero un modelo DDPM para generar rostros de anime.

Ejemplo:
python train.py --data-dir data/animefacedataset --epochs 20 --batch-size 32
"""
from __future__ import annotations

import argparse
import copy
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import AnimeFacesDataset
from diffusion import Diffusion
from unet import AnimeUNet


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento DDPM de rostros de anime")
    parser.add_argument("--data-dir", type=str, default="data/animefacedataset",
                        help="Carpeta donde se descomprimió el dataset")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--resume", type=str, default="",
                        help="Ruta a latest.pt para continuar entrenamiento")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.995) -> None:
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(model_param.data, alpha=1.0 - decay)
        for ema_buffer, model_buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(model_buffer)


def save_checkpoint(path: Path, epoch: int, model: nn.Module, ema_model: nn.Module,
                    optimizer: torch.optim.Optimizer, args, best_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "ema_state": ema_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "timesteps": args.timesteps,
        "base_channels": args.base_channels,
        "image_size": 64,
        "best_loss": best_loss,
        "config": vars(args),
    }, path)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs y batch-size deben ser mayores que cero.")
    if args.base_channels % 8 != 0:
        raise ValueError("base-channels debe ser múltiplo de 8 (por ejemplo 32 o 64).")
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True


    dataset = AnimeFacesDataset(args.data_dir, image_size=64)
    print(f"Imágenes válidas encontradas: {len(dataset):,}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    model = AnimeUNet(image_channels=3, base_channels=args.base_channels).to(device)
    ema_model = copy.deepcopy(model).to(device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    diffusion = Diffusion(args.timesteps, device)
    loss_function = nn.MSELoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir.parent / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir.parent / "training_log.csv"
    start_epoch = 0
    best_loss = float("inf")

    if args.resume:
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No existe el checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        ema_model.load_state_dict(checkpoint.get("ema_state", checkpoint["model_state"]))
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_loss = float(checkpoint.get("best_loss", float("inf")))
        print(f"Continuando desde la época {start_epoch}.")

    write_header = not log_path.exists() or start_epoch == 0
    log_file = log_path.open("a" if not write_header else "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(["epoch", "average_mse_loss", "learning_rate", "seconds"])

    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            epoch_start = time.time()
            total_loss = 0.0
            progress = tqdm(loader, desc=f"Época {epoch + 1}/{args.epochs}")

            for real_images in progress:
                real_images = real_images.to(device, non_blocking=True)
                batch = real_images.shape[0]
                timesteps = torch.randint(0, args.timesteps, (batch,), device=device)
                noise = torch.randn_like(real_images)
                noisy_images = diffusion.add_noise(real_images, noise, timesteps)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    predicted_noise = model(noisy_images, timesteps)
                    loss = loss_function(predicted_noise, noise)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                update_ema(ema_model, model)

                total_loss += loss.item() * batch
                progress.set_postfix(loss=f"{loss.item():.5f}")

            average_loss = total_loss / len(dataset)
            elapsed = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Época {epoch + 1}: pérdida MSE={average_loss:.6f} | tiempo={elapsed:.1f}s")
            log_writer.writerow([epoch + 1, f"{average_loss:.8f}", f"{current_lr:.8f}", f"{elapsed:.1f}"])
            log_file.flush()

            if average_loss < best_loss:
                best_loss = average_loss
                save_checkpoint(output_dir / "best.pt", epoch + 1, model, ema_model,
                                optimizer, args, best_loss)

            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(output_dir / "latest.pt", epoch + 1, model, ema_model,
                                optimizer, args, best_loss)

            if args.sample_every > 0 and (epoch + 1) % args.sample_every == 0:
                ema_model.eval()
                with torch.no_grad():
                    previews = diffusion.sample_ddim(
                        ema_model, number_of_images=16, sampling_steps=min(50, args.timesteps)
                    )
                save_image(
                    (previews + 1) / 2,
                    samples_dir / f"epoch_{epoch + 1:03d}.png",
                    nrow=4,
                )
                print(f"Muestra guardada en: {samples_dir / f'epoch_{epoch + 1:03d}.png'}")

        # Guarda siempre un checkpoint final para que no se pierda la última época.
        save_checkpoint(output_dir / "final.pt", args.epochs, model, ema_model,
                        optimizer, args, best_loss)
        print("\nEntrenamiento terminado.")
        print(f"Mejor pérdida MSE registrada: {best_loss:.6f}")
        print(f"Checkpoints: {output_dir.resolve()}")
        print(f"Registro CSV: {log_path.resolve()}")
        print(f"Muestras: {samples_dir.resolve()}")
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
