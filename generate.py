"""Genera nuevos rostros de anime desde un checkpoint entrenado.

Ejemplo:
python generate.py --checkpoint checkpoints/best.pt --num-images 16 --sampling-steps 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from diffusion import Diffusion
from unet import AnimeUNet


def parse_args():
    parser = argparse.ArgumentParser(description="Generación de rostros anime con DDPM/DDIM")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--sampling-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--eta", type=float, default=0.0,
                        help="0 = DDIM determinista; valores mayores introducen aleatoriedad")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_images < 1:
        raise ValueError("num-images debe ser mayor que cero.")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No se encontró {checkpoint_path}. Primero ejecuta train.py y comprueba el checkpoint."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de generación: {device}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    timesteps = int(checkpoint.get("timesteps", 1000))
    base_channels = int(checkpoint.get("base_channels", 64))
    image_size = int(checkpoint.get("image_size", 64))

    model = AnimeUNet(image_channels=3, base_channels=base_channels).to(device)
    weights = checkpoint.get("ema_state", checkpoint["model_state"])
    model.load_state_dict(weights)
    model.eval()

    diffusion = Diffusion(timesteps=timesteps, device=device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    generated = diffusion.sample_ddim(
        model=model,
        number_of_images=args.num_images,
        image_size=image_size,
        sampling_steps=args.sampling_steps,
        eta=args.eta,
    )

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    normalized = (generated.cpu() + 1.0) / 2.0

    for index, image in enumerate(normalized):
        save_image(image, images_dir / f"anime_face_{index + 1:03d}.png")

    grid_path = output_dir / "anime_faces_grid.png"
    save_image(normalized, grid_path, nrow=min(4, args.num_images))
    print(f"Se generaron {args.num_images} imágenes.")
    print(f"Cuadrícula: {grid_path.resolve()}")
    print(f"Imágenes individuales: {images_dir.resolve()}")


if __name__ == "__main__":
    main()
