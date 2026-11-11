"""U-Net para predecir el ruido en un modelo de difusión DDPM."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Convierte el paso de difusión t en una representación sinusoidal."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        if half == 0:
            return timesteps.float().unsqueeze(1)
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(nn.Module):
    """Bloque residual condicionado por el paso temporal."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("Los canales deben ser divisibles por el número de grupos.")

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        h = self.conv1(F.silu(self.norm1(x)))
        time_term = self.time_projection(F.silu(time_embedding)).unsqueeze(-1).unsqueeze(-1)
        h = h + time_term
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + residual


class SelfAttention2D(nn.Module):
    """Atención propia espacial para que el modelo relacione distintas zonas del rostro."""

    def __init__(self, channels: int, groups: int = 8, heads: int = 4) -> None:
        super().__init__()
        if channels % groups != 0 or channels % heads != 0:
            raise ValueError("Canales incompatibles con groups o heads.")
        self.norm = nn.GroupNorm(groups, channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels, num_heads=heads, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        normalized = self.norm(x).reshape(batch, channels, height * width).transpose(1, 2)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        attended = attended.transpose(1, 2).reshape(batch, channels, height, width)
        return x + attended


class AnimeUNet(nn.Module):
    """U-Net de 3 canales para imágenes RGB de 64x64."""

    def __init__(self, image_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        b = base_channels
        time_dim = b * 4
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(b),
            nn.Linear(b, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.stem = nn.Conv2d(image_channels, b, kernel_size=3, padding=1)

        # Encoder: 64x64 -> 32x32 -> 16x16 -> 8x8
        self.down1a = ResidualBlock(b, b, time_dim)
        self.down1b = ResidualBlock(b, b, time_dim)
        self.downsample1 = nn.Conv2d(b, b * 2, kernel_size=4, stride=2, padding=1)

        self.down2a = ResidualBlock(b * 2, b * 2, time_dim)
        self.down2b = ResidualBlock(b * 2, b * 2, time_dim)
        self.downsample2 = nn.Conv2d(b * 2, b * 4, kernel_size=4, stride=2, padding=1)

        self.down3a = ResidualBlock(b * 4, b * 4, time_dim)
        self.down3attn = SelfAttention2D(b * 4)
        self.down3b = ResidualBlock(b * 4, b * 4, time_dim)
        self.downsample3 = nn.Conv2d(b * 4, b * 4, kernel_size=4, stride=2, padding=1)

        # Bottleneck a 8x8.
        self.mid1 = ResidualBlock(b * 4, b * 4, time_dim)
        self.midattn = SelfAttention2D(b * 4)
        self.mid2 = ResidualBlock(b * 4, b * 4, time_dim)

        # Decoder con conexiones skip del encoder.
        self.upsample3 = nn.Conv2d(b * 4, b * 4, kernel_size=3, padding=1)
        self.up3a = ResidualBlock(b * 8, b * 4, time_dim)
        self.up3attn = SelfAttention2D(b * 4)
        self.up3b = ResidualBlock(b * 4, b * 4, time_dim)

        self.upsample2 = nn.Conv2d(b * 4, b * 2, kernel_size=3, padding=1)
        self.up2a = ResidualBlock(b * 4, b * 2, time_dim)
        self.up2b = ResidualBlock(b * 2, b * 2, time_dim)

        self.upsample1 = nn.Conv2d(b * 2, b, kernel_size=3, padding=1)
        self.up1a = ResidualBlock(b * 2, b, time_dim)
        self.up1b = ResidualBlock(b, b, time_dim)

        self.output = nn.Sequential(
            nn.GroupNorm(8, b),
            nn.SiLU(),
            nn.Conv2d(b, image_channels, kernel_size=3, padding=1),
        )

    @staticmethod
    def _upsample(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=target.shape[-2:], mode="nearest")

    def forward(self, image: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("La entrada debe tener forma [batch, 3, alto, ancho].")
        if image.shape[-2:] != (64, 64):
            raise ValueError("Este proyecto está configurado para imágenes de 64x64.")

        t = self.time_embedding(timesteps)
        x = self.stem(image)

        s1 = self.down1b(self.down1a(x, t), t)
        x = self.downsample1(s1)
        s2 = self.down2b(self.down2a(x, t), t)
        x = self.downsample2(s2)
        x = self.down3a(x, t)
        x = self.down3attn(x)
        s3 = self.down3b(x, t)
        x = self.downsample3(s3)

        x = self.mid1(x, t)
        x = self.midattn(x)
        x = self.mid2(x, t)

        x = self._upsample(x, s3)
        x = self.upsample3(x)
        x = self.up3a(torch.cat([x, s3], dim=1), t)
        x = self.up3attn(x)
        x = self.up3b(x, t)

        x = self._upsample(x, s2)
        x = self.upsample2(x)
        x = self.up2a(torch.cat([x, s2], dim=1), t)
        x = self.up2b(x, t)

        x = self._upsample(x, s1)
        x = self.upsample1(x)
        x = self.up1a(torch.cat([x, s1], dim=1), t)
        x = self.up1b(x, t)
        return self.output(x)
