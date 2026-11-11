"""Carga recursiva del Anime Face Dataset desde una carpeta de imágenes."""
from __future__ import annotations

from pathlib import Path
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class AnimeFacesDataset(Dataset):
    def __init__(self, data_dir: str | Path, image_size: int = 64) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"No existe la carpeta del dataset: {self.data_dir}\n"
                "Descarga y descomprime Anime Face Dataset, y vuelve a ejecutar con --data-dir."
            )

        candidates = sorted(
            p for p in self.data_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        # Ignora archivos de imagen dañados para evitar que detengan el entrenamiento.
        self.image_paths: list[Path] = []
        for path in candidates:
            try:
                with Image.open(path) as image:
                    image.verify()
                self.image_paths.append(path)
            except (OSError, UnidentifiedImageError, ValueError):
                continue

        if not self.image_paths:
            raise RuntimeError(
                f"No se encontraron imágenes JPG/PNG dentro de {self.data_dir}. "
                "Comprueba que el ZIP esté descomprimido."
            )

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                return self.transform(image)
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"No se pudo leer la imagen: {path}") from exc
