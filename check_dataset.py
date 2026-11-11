"""Comprueba que el dataset esté descomprimido y crea una vista previa."""
from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, UnidentifiedImageError

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verificar Anime Face Dataset")
    parser.add_argument("--data-dir", default="data/animefacedataset")
    parser.add_argument("--preview", default="dataset_preview.png")
    args = parser.parse_args()

    root = Path(args.data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"No existe {root}. Descomprime el ZIP dentro de data/animefacedataset."
        )

    paths = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    print(f"Carpeta revisada: {root}")
    print(f"Archivos con extensión de imagen encontrados: {len(paths):,}")
    if not paths:
        raise RuntimeError("No hay imágenes. Revisa que el ZIP esté descomprimido.")

    valid_images = []
    for path in paths:
        try:
            with Image.open(path) as image:
                valid_images.append((path, image.convert("RGB").copy()))
        except (OSError, UnidentifiedImageError):
            continue
        if len(valid_images) >= 16:
            break

    if not valid_images:
        raise RuntimeError("Se encontraron nombres de imágenes, pero ninguna pudo abrirse.")

    cell = 128
    margin = 12
    columns = 4
    rows = (len(valid_images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell, rows * (cell + 28)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (path, image) in enumerate(valid_images):
        image.thumbnail((cell - 8, cell - 8))
        x = (index % columns) * cell + (cell - image.width) // 2
        y = (index // columns) * (cell + 28) + (cell - image.height) // 2
        canvas.paste(image, (x, y))
        draw.text(((index % columns) * cell + 4,
                   (index // columns) * (cell + 28) + cell), path.name[:20], fill="black")

    preview_path = Path(args.preview).resolve()
    canvas.save(preview_path)
    print(f"Vista previa creada con {len(valid_images)} imágenes: {preview_path}")
    print("Si los rostros se ven correctamente, el dataset está listo para entrenar.")


if __name__ == "__main__":
    main()
