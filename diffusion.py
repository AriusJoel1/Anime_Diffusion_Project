"""Proceso DDPM de entrenamiento y muestreo DDIM para generación de imágenes."""
from __future__ import annotations

import torch


class Diffusion:
    def __init__(self, timesteps: int = 1000, device: str | torch.device = "cpu") -> None:
        if timesteps < 2:
            raise ValueError("timesteps debe ser mayor o igual que 2.")
        self.timesteps = timesteps
        self.device = torch.device(device)

        # Programación lineal de beta, igual al esquema explicado en clase.
        self.betas = torch.linspace(1e-4, 0.02, timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(
        self, clean_images: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Ecuación directa q(x_t | x_0): combina imagen limpia y ruido."""
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar) * clean_images + torch.sqrt(1.0 - alpha_bar) * noise

    @torch.no_grad()
    def sample_ddim(
        self,
        model: torch.nn.Module,
        number_of_images: int,
        image_size: int = 64,
        channels: int = 3,
        sampling_steps: int = 100,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Genera imágenes con DDIM usando menos pasos que DDPM completo."""
        if not 1 <= sampling_steps <= self.timesteps:
            raise ValueError(f"sampling_steps debe estar entre 1 y {self.timesteps}.")
        model.eval()
        x = torch.randn(
            number_of_images, channels, image_size, image_size, device=self.device
        )
        time_grid = torch.linspace(
            self.timesteps - 1, 0, sampling_steps, device=self.device
        ).long()

        for index, current_t in enumerate(time_grid):
            t_batch = torch.full(
                (number_of_images,), int(current_t.item()), device=self.device, dtype=torch.long
            )
            predicted_noise = model(x, t_batch)

            alpha_bar_t = self.alpha_bars[current_t]
            if index + 1 < len(time_grid):
                next_t = time_grid[index + 1]
                alpha_bar_previous = self.alpha_bars[next_t]
            else:
                alpha_bar_previous = torch.tensor(1.0, device=self.device)

            predicted_clean = (
                x - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise
            ) / torch.sqrt(alpha_bar_t)
            predicted_clean = predicted_clean.clamp(-1.0, 1.0)

            # DDIM: eta=0 produce un recorrido determinista.
            sigma = eta * torch.sqrt(
                ((1.0 - alpha_bar_previous) / (1.0 - alpha_bar_t)
                 * (1.0 - alpha_bar_t / alpha_bar_previous)).clamp(min=0.0)
            )
            direction = torch.sqrt(
                (1.0 - alpha_bar_previous - sigma**2).clamp(min=0.0)
            ) * predicted_noise
            x = torch.sqrt(alpha_bar_previous) * predicted_clean + direction
            if eta > 0 and index + 1 < len(time_grid):
                x = x + sigma * torch.randn_like(x)

        return x.clamp(-1.0, 1.0)
