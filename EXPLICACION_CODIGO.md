# Explicación del código: Modelo de Difusión para rostros de anime

Este documento explica qué hace cada archivo del proyecto, cuáles son sus entradas y salidas, cómo funciona el algoritmo de difusión (DDPM) y cómo se genera una imagen con muestreo DDIM. El proyecto entrena una U-Net desde cero para predecir ruido y así aprender a generar rostros de anime de **64 × 64 píxeles en RGB**.

---

## 1. Visión general

El flujo del proyecto es el siguiente:

```text
Dataset de imágenes (JPG/PNG)
        │
        ▼
dataset.py  ──►  tensor [3, 64, 64] normalizado a [-1, 1]
        │
        ▼
train.py  ──►  agrega ruido a la imagen (diffusion.add_noise)
        │      entrena la U-Net para predecir ese ruido
        ▼
checkpoints/*.pt  (modelo + EMA + optimizador)
        │
        ▼
generate.py  ──►  parte de ruido aleatorio y aplica DDIM
        │
        ▼
results/anime_faces_grid.png + results/images/*.png
```

La idea central de un modelo de difusión es:

1. **Proceso directo (forward):** a una imagen real se le agrega ruido gaussiano poco a poco hasta que queda irreconocible.
2. **Entrenamiento:** la red aprende a estimar el ruido que se añadió en cada paso.
3. **Proceso inverso (reverse):** partiendo de ruido puro, la red va quitando ruido paso a paso y genera una imagen nueva.

---

## 2. Estructura de archivos

| Archivo | Función |
|---|---|
| `check_dataset.py` | Verifica que el dataset exista, cuenta imágenes y crea una vista previa. |
| `dataset.py` | Carga, valida y preprocesa las imágenes para el entrenamiento. |
| `unet.py` | Define la arquitectura U-Net condicionada por el paso temporal. |
| `diffusion.py` | Define el calendario de ruido, la ecuación directa y el muestreo DDIM. |
| `train.py` | Entrena la U-Net con pérdida MSE, AdamW, AMP y pesos EMA. |
| `generate.py` | Carga un checkpoint y genera imágenes nuevas. |
| `requirements.txt` | Dependencias del proyecto. |
| `1_entrenar.bat` / `2_generar.bat` | Atajos de Windows para entrenar y generar. |

---

## 3. Entradas y salidas del proyecto

### 3.1 Entrada del entrenamiento

- Carpeta con imágenes del **Anime Face Dataset** (por defecto `data/animefacedataset`).
- Formatos aceptados: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.
- No importa si las imágenes están en subcarpetas: la búsqueda es recursiva.
- Cada imagen se convierte a RGB y se redimensiona a **64 × 64**.

### 3.2 Entrada de la generación

- Un checkpoint entrenado: `checkpoints/best.pt`, `latest.pt` o `final.pt`.
- Parámetros de generación: número de imágenes, pasos de muestreo, semilla y `eta`.

### 3.3 Salidas

| Salida | Descripción |
|---|---|
| `checkpoints/best.pt` | Mejor época según la pérdida MSE promedio. |
| `checkpoints/latest.pt` | Último estado guardado, sirve para continuar entrenando. |
| `checkpoints/final.pt` | Estado final de la última época solicitada. |
| `training_log.csv` | Pérdida MSE, learning rate y segundos por época. |
| `samples/epoch_XXX.png` | Cuadrícula de 16 imágenes generadas durante el entrenamiento. |
| `results/anime_faces_grid.png` | Cuadrícula con las imágenes generadas. |
| `results/images/anime_face_XXX.png` | Cada imagen generada por separado. |
| `dataset_preview.png` | Vista previa para validar el dataset. |

Cada checkpoint contiene:

```python
{
    "epoch": ...,            # época alcanzada
    "model_state": ...,      # pesos de la U-Net
    "ema_state": ...,        # pesos EMA (se usan para generar)
    "optimizer_state": ...,  # estado de AdamW (para reanudar)
    "timesteps": 1000,
    "base_channels": 64,
    "image_size": 64,
    "best_loss": ...,
    "config": {...},         # argumentos usados en el entrenamiento
}
```

---

## 4. `dataset.py`: carga y preprocesamiento

### 4.1 Qué hace

1. Busca recursivamente todos los archivos con extensión de imagen dentro de `data_dir`.
2. Valida cada imagen con `PIL.Image.verify()` y descarta las dañadas.
3. Define las transformaciones:

```python
transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
```

### 4.2 Entrada y salida

- **Entrada:** ruta de una imagen.
- **Salida:** tensor de forma `[3, 64, 64]` con valores en `[-1, 1]`.

La normalización con media 0.5 y desviación 0.5 transforma los píxeles de `[0, 1]` a `[-1, 1]`. Este rango es el que espera la U-Net y el que el proceso de difusión utiliza. Al guardar imágenes, se invierte con `(x + 1) / 2`.

El **volteo horizontal aleatorio** es una augmentation que duplica la variedad de rostros y reduce el sobreajuste.

---

## 5. `unet.py`: arquitectura del modelo

La red recibe una imagen ruidosa y el paso temporal `t`, y devuelve una estimación del ruido que contiene esa imagen.

### 5.1 Entrada y salida

- **Entrada:**
  - `image`: tensor `[B, 3, 64, 64]`
  - `timesteps`: tensor `[B]` con enteros entre `0` y `999`
- **Salida:** tensor `[B, 3, 64, 64]` con el ruido predicho.

### 5.2 Componentes

**a) Embedding temporal sinusoidal (`SinusoidalTimeEmbedding`)**

Convierte el paso `t` en un vector que representa su "posición" en el calendario de ruido, de forma análoga a los embeddings posicionales de los Transformers:

```python
frequencies = exp(-log(10000) * arange(half) / (half - 1))
angles = t * frequencies
embedding = concat([sin(angles), cos(angles)])
```

Luego pasa por dos capas lineales con activación SiLU, produciendo un vector de dimensión `time_dim = 4 × base_channels`.

**b) Bloques residuales (`ResidualBlock`)**

Cada bloque aplica:

```text
GroupNorm → SiLU → Conv2d → (+ embedding temporal) → GroupNorm → SiLU → Dropout → Conv2d → (+ shortcut)
```

El embedding temporal se suma a los mapas de características para que la red sepa en qué nivel de ruido está trabajando.

**c) Atención espacial (`SelfAttention2D`)**

Se aplica en las resoluciones 16 × 16 y 8 × 8. Permite que el modelo relacione zonas distantes del rostro (por ejemplo, ojos y boca) mediante `MultiheadAttention` con 4 cabezas.

**d) Codificador (encoder)**

| Etapa | Resolución | Canales |
|---|---:|---:|
| Stem | 64 × 64 | `b` |
| Down 1 | 64 × 64 → 32 × 32 | `b` → `2b` |
| Down 2 | 32 × 32 → 16 × 16 | `2b` → `4b` |
| Down 3 | 16 × 16 → 8 × 8 | `4b` |

**e) Cuello de botella (bottleneck)**

A 8 × 8 con canales `4b`: bloque residual → atención → bloque residual.

**f) Decodificador (decoder)**

Sube la resolución con interpolación nearest y concatena las conexiones skip del encoder:

```text
8×8 → 16×16 → 32×32 → 64×64
```

Las conexiones skip permiten recuperar detalles finos que se pierden al comprimir la imagen.

**g) Salida**

`GroupNorm → SiLU → Conv2d(3)` devuelve los 3 canales RGB del ruido predicho.

Con `base_channels = 64`, la red tiene aproximadamente 20–25 millones de parámetros.

---

## 6. `diffusion.py`: el algoritmo de difusión

### 6.1 Calendario de ruido

El proyecto usa una **programación lineal de beta**, igual al esquema DDPM original:

```python
betas = linspace(1e-4, 0.02, 1000)
alphas = 1 - betas
alpha_bars = cumprod(alphas)
```

Donde:

- `β_t`: cantidad de ruido añadida en el paso `t`.
- `α_t = 1 - β_t`.
- `ᾱ_t = ∏_{s=1}^{t} α_s`: producto acumulado. Va de ≈1 al inicio a ≈0 al final.

### 6.2 Proceso directo (forward)

Agregar ruido paso a paso se puede resumir en una sola ecuación, sin iterar:

```text
q(x_t | x_0) = N( sqrt(ᾱ_t) · x_0 , (1 - ᾱ_t) · I )
```

Es decir:

```text
x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 - ᾱ_t) · ε ,   ε ~ N(0, I)
```

Esto es exactamente lo que implementa `add_noise`:

```python
alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1, 1)
return sqrt(alpha_bar) * clean_images + sqrt(1 - alpha_bar) * noise
```

- Al inicio (`t ≈ 0`): `ᾱ_t ≈ 1`, la imagen casi no tiene ruido.
- Al final (`t = 999`): `ᾱ_t ≈ 0`, la imagen es prácticamente ruido puro.

La ventaja de esta forma cerrada es que durante el entrenamiento se puede muestrear cualquier `t` directamente, sin simular todos los pasos anteriores.

### 6.3 Objetivo de entrenamiento

La red `ε_θ(x_t, t)` aprende a predecir el ruido `ε` que se añadió:

```text
L = E_{x_0, ε, t} [ || ε - ε_θ( sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε , t ) ||² ]
```

En `train.py` se implementa con:

```python
timesteps = torch.randint(0, 1000, (batch,), device=device)
noise = torch.randn_like(real_images)
noisy_images = diffusion.add_noise(real_images, noise, timesteps)
predicted_noise = model(noisy_images, timesteps)
loss = nn.MSELoss()(predicted_noise, noise)
```

Predecir el ruido es equivalente, en esperanza, a predecir la imagen limpia o la puntuación (score), pero resulta más estable numéricamente.

### 6.4 Proceso inverso con DDIM

Para generar, `sample_ddim` parte de ruido puro `x_T ~ N(0, I)` y recorre los pasos en orden descendente. En lugar de usar los 1000 pasos completos, utiliza un subconjunto de `sampling_steps` pasos (100 por defecto).

En cada paso:

**1. Predecir el ruido:**

```text
ε̂ = ε_θ(x_t, t)
```

**2. Estimar la imagen limpia a partir de `x_t`:**

```text
x̂_0 = (x_t - sqrt(1 - ᾱ_t) · ε̂) / sqrt(ᾱ_t)
```

Se recorta a `[-1, 1]` con `clamp` para evitar valores fuera de rango.

**3. Calcular el siguiente estado con la actualización DDIM:**

```text
x_{t-1} = sqrt(ᾱ_{t-1}) · x̂_0 + sqrt(1 - ᾱ_{t-1} - σ²) · ε̂ + σ · z
```

Donde:

- `σ = η · sqrt( (1-ᾱ_{t-1})/(1-ᾱ_t) · (1 - ᾱ_t/ᾱ_{t-1}) )`
- `z ~ N(0, I)` solo se añade si `η > 0`.

Con `η = 0` (valor por defecto), el proceso es **determinista**: dada la misma semilla, se obtiene siempre la misma imagen. Con `η > 0` se recupera parte del comportamiento estocástico de DDPM.

En el último paso se usa `ᾱ_{t-1} = 1`, de modo que la salida final es directamente `x̂_0`.

**Ventaja de DDIM:** permite generar con 50–100 pasos en lugar de 1000, reduciendo el tiempo de inferencia sin perder demasiada calidad.

---

## 7. `train.py`: entrenamiento

### 7.1 Flujo por lote

```text
1. Cargar lote de imágenes reales        [B, 3, 64, 64]
2. Muestrear t uniforme en [0, 999]      [B]
3. Muestrear ruido ε ~ N(0, I)           [B, 3, 64, 64]
4. Construir x_t = add_noise(x_0, ε, t)  [B, 3, 64, 64]
5. Predecir ε̂ = U-Net(x_t, t)           [B, 3, 64, 64]
6. Calcular MSE(ε̂, ε)
7. Retropropagar con AdamW + AMP
8. Actualizar pesos EMA
```

### 7.2 Componentes del entrenamiento

| Componente | Valor / función |
|---|---|
| Optimizador | AdamW, `lr = 2e-4`, `weight_decay = 1e-4` |
| Pérdida | MSE entre ruido real y predicho |
| Precisión mixta | `torch.amp` en GPU (ahorra memoria y acelera) |
| Recorte de gradiente | `max_norm = 1.0` (estabiliza el entrenamiento) |
| EMA | `decay = 0.995` sobre los pesos del modelo |

**¿Qué es EMA?** Mantiene una copia "suavizada" de los pesos:

```text
θ_ema = 0.995 · θ_ema + 0.005 · θ_modelo
```

Esta copia promedia el entrenamiento y produce muestras más estables y de mejor calidad. Por eso `generate.py` usa `ema_state` en lugar de `model_state`.

### 7.3 Guardado y registro

- Guarda `best.pt` cuando la pérdida promedio de la época mejora.
- Guarda `latest.pt` cada `--save-every` épocas.
- Guarda `final.pt` al terminar.
- Cada `--sample-every` épocas genera una cuadrícula de 16 imágenes con DDIM (50 pasos) para inspección visual.
- `training_log.csv` registra `epoch`, `average_mse_loss`, `learning_rate` y `seconds`.

### 7.4 Parámetros principales

| Parámetro | Valor por defecto | Descripción |
|---|---:|---|
| `--data-dir` | `data/animefacedataset` | Carpeta del dataset. |
| `--epochs` | 20 | Épocas de entrenamiento. |
| `--batch-size` | 32 | Imágenes por lote. |
| `--learning-rate` | `2e-4` | Tasa de aprendizaje. |
| `--timesteps` | 1000 | Pasos del proceso de difusión. |
| `--base-channels` | 64 | Ancho base de la U-Net (múltiplo de 8). |
| `--num-workers` | 2 | Procesos de carga de datos. |
| `--save-every` | 1 | Frecuencia de guardado de `latest.pt`. |
| `--sample-every` | 5 | Frecuencia de muestras visuales. |
| `--resume` | `""` | Checkpoint para continuar entrenando. |
| `--seed` | 42 | Semilla de reproducibilidad. |

---

## 8. `generate.py`: generación de imágenes

### 8.1 Flujo

```text
1. Cargar checkpoint (timesteps, base_channels, image_size)
2. Construir la U-Net y cargar ema_state
3. Crear ruido x_T ~ N(0, I) de forma [N, 3, 64, 64]
4. Aplicar sample_ddim con los pasos indicados
5. Desnormalizar a [0, 1] con (x + 1) / 2
6. Guardar imágenes individuales y una cuadrícula
```

### 8.2 Parámetros

| Parámetro | Valor por defecto | Descripción |
|---|---:|---|
| `--checkpoint` | `checkpoints/best.pt` | Modelo entrenado a usar. |
| `--output-dir` | `results` | Carpeta de salida. |
| `--num-images` | 16 | Cantidad de imágenes a generar. |
| `--sampling-steps` | 100 | Pasos DDIM. |
| `--seed` | 123 | Semilla para reproducibilidad. |
| `--eta` | 0.0 | 0 = determinista; >0 = añade estocasticidad. |

La cuadrícula usa `nrow = min(4, num_images)`, por lo que se muestran 4 imágenes por fila.

---

## 9. `check_dataset.py`: verificación del dataset

- **Entrada:** `--data-dir` y `--preview`.
- **Salida:** `dataset_preview.png` con hasta 16 miniaturas y sus nombres.
- Cuenta los archivos con extensión válida y valida que puedan abrirse.
- Sirve para confirmar que la descarga y descompresión del dataset fueron correctas antes de entrenar.

---

## 10. Cómo ejecutar el proyecto

### 10.1 Verificar el dataset

```bash
python check_dataset.py --data-dir data/animefacedataset
```

### 10.2 Entrenar

```bash
python train.py --data-dir data/animefacedataset --epochs 20 --batch-size 32
```

Prueba rápida de una época:

```bash
python train.py --data-dir data/animefacedataset --epochs 1 --batch-size 16 --base-channels 32 --num-workers 0
```

Continuar desde un checkpoint:

```bash
python train.py --data-dir data/animefacedataset --epochs 40 --batch-size 32 --resume checkpoints/latest.pt
```

### 10.3 Generar imágenes

```bash
python generate.py --checkpoint checkpoints/best.pt --num-images 16 --sampling-steps 100
```

También se puede usar `1_entrenar.bat` y `2_generar.bat` en Windows.

---

## 11. Resumen del algoritmo

```text
ENTRENAMIENTO (DDPM)
────────────────────
Para cada imagen x_0:
  1. Elegir t ~ Uniforme{0, ..., T-1}
  2. Muestrear ε ~ N(0, I)
  3. Construir x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε
  4. Predecir ε̂ = ε_θ(x_t, t)
  5. Minimizar ||ε - ε̂||²
  6. Actualizar θ con AdamW y actualizar θ_ema

GENERACIÓN (DDIM)
─────────────────
  1. x_T ~ N(0, I)
  2. Para t = T-1, ..., 0 (subconjunto de pasos):
       ε̂ = ε_θ(x_t, t)
       x̂_0 = (x_t - sqrt(1-ᾱ_t)·ε̂) / sqrt(ᾱ_t)
       x̂_0 = clamp(x̂_0, -1, 1)
       x_{t-1} = sqrt(ᾱ_{t-1})·x̂_0 + sqrt(1-ᾱ_{t-1}-σ²)·ε̂ + σ·z
  3. Devolver x_0
```

---

## 12. Limitaciones y notas

- **No es condicional:** el modelo no recibe texto ni etiquetas; genera rostros aleatorios según lo aprendido del dataset.
- **Resolución fija:** 64 × 64. La U-Net valida explícitamente esta forma.
- **Entrenamiento desde cero:** no se usan pesos preentrenados. La calidad depende mucho del número de épocas, la GPU y el tamaño del dataset.
- **20 épocas es un punto de partida:** para resultados más nítidos suelen ser necesarias más épocas y mayor capacidad.
- **DDIM es una aproximación:** con pocos pasos se pierde algo de calidad frente a usar los 1000 pasos completos.
- **La semilla controla la reproducibilidad:** con la misma semilla, `eta = 0` y el mismo checkpoint, se obtienen las mismas imágenes.
- **Uso académico:** las imágenes generadas deben provenir de ejecutar el proyecto; no se incluyen resultados inventados ni pesos preentrenados.
