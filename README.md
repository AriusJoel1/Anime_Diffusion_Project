# Generación de rostros de anime con Modelos de Difusión (DDPM)

Proyecto académico basado en el flujo de difusión visto en clase: se agrega ruido progresivamente a imágenes reales y se entrena una U-Net para predecir ese ruido. Para generar imágenes, se parte de ruido aleatorio y se aplica el proceso inverso con muestreo DDIM.

## 1. Requisitos

- Python 3.10 o 3.11 recomendado.
- GPU NVIDIA con CUDA recomendada. También funciona en CPU, pero el entrenamiento será mucho más lento.
- Anime Face Dataset de Kaggle: <https://www.kaggle.com/datasets/splcher/animefacedataset>.

## 2. Estructura esperada

Descomprime el dataset dentro de `data/animefacedataset`. El cargador busca imágenes recursivamente, así que también reconoce una subcarpeta `images` u otras carpetas internas.

```text
Anime_Diffusion_Project/
├── data/
│   └── animefacedataset/
│       └── images/            # o las imágenes directamente
├── checkpoints/
├── samples/
├── results/
├── dataset.py
├── diffusion.py
├── unet.py
├── train.py
├── generate.py
└── requirements.txt
```

## 3. Instalación

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Instala PyTorch con la versión de CUDA que corresponda a tu equipo desde <https://pytorch.org/get-started/locally/>. Si no tienes GPU NVIDIA, `pip install -r requirements.txt` instalará la versión disponible para tu plataforma y se usará CPU.

### Google Colab (recomendado si tu computadora no tiene GPU)

1. En Colab, selecciona **Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU**.
2. Sube esta carpeta del proyecto o clona/descarga sus archivos en `/content/Anime_Diffusion_Project`.
3. Ejecuta en una celda:

```python
%cd /content/Anime_Diffusion_Project
!pip install -r requirements.txt
```

## 4. Descargar el dataset

### Opción A: navegador (más sencilla)

1. Abre <https://www.kaggle.com/datasets/splcher/animefacedataset> e inicia sesión en Kaggle.
2. Pulsa **Download** para bajar el archivo ZIP.
3. Crea la carpeta `data/animefacedataset` dentro del proyecto.
4. Extrae el contenido del ZIP allí. No cambies ni renombres las imágenes.
5. Verifica que dentro de esa carpeta haya imágenes `.jpg`, `.jpeg` o `.png`. Es posible que estén dentro de una subcarpeta llamada `images`.

### Opción B: Kaggle API (útil en Colab)

1. En Kaggle, entra a tu perfil > **Settings** > **API** > **Create New Token**. Se descargará `kaggle.json`.
2. En una celda de Colab ejecuta:

```python
!pip install -q kaggle
from google.colab import files
uploaded = files.upload()  # Selecciona kaggle.json
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!mkdir -p /content/Anime_Diffusion_Project/data/animefacedataset
!kaggle datasets download -d splcher/animefacedataset \
  -p /content/Anime_Diffusion_Project/data/animefacedataset --unzip
```

No compartas `kaggle.json`: contiene credenciales de tu cuenta. Si Kaggle solicita aceptar las condiciones del dataset, hazlo desde su página.

## 5. Verificar el dataset

Antes de entrenar, comprueba que el programa encuentre imágenes y crea una vista previa:

```bash
python check_dataset.py --data-dir data/animefacedataset
```

Se creará `dataset_preview.png`. Revisa que se vean rostros de anime y que el recuento sea razonable.

## 6. Entrenamiento

Desde la carpeta del proyecto:

```bash
python train.py --data-dir data/animefacedataset --epochs 20 --batch-size 32
```

En Colab, antepón `!` al comando:

```python
!python train.py --data-dir data/animefacedataset --epochs 20 --batch-size 32
```

Antes de iniciar todo el entrenamiento, es recomendable probar una época:

```bash
python train.py --data-dir data/animefacedataset --epochs 1 --batch-size 16 --base-channels 32 --num-workers 0
```

Si funciona, inicia el entrenamiento definitivo. En una GPU con memoria limitada, reduce el batch size a 16 u 8. Si el equipo no soporta el tamaño del modelo, puedes reducir `--base-channels` a 32. Un entrenamiento de una sola época sirve para validar el pipeline, pero normalmente no basta para obtener resultados de buena calidad.

El entrenamiento genera:

- `checkpoints/latest.pt`: estado más reciente para continuar el entrenamiento.
- `checkpoints/best.pt`: checkpoint de la época con menor pérdida promedio.
- `checkpoints/final.pt`: checkpoint de la última época solicitada.
- `training_log.csv`: pérdida MSE, tasa de aprendizaje y tiempo por época.
- `samples/epoch_XXX.png`: muestras de rostros generadas durante el entrenamiento.

Para continuar desde el último checkpoint:

```bash
python train.py --data-dir data/animefacedataset --epochs 40 --batch-size 32 --resume checkpoints/latest.pt
```

En ese ejemplo se solicita llegar a la época 40 en total; no son 40 épocas adicionales.

## 7. Generación de rostros

Después de entrenar:

```bash
python generate.py --checkpoint checkpoints/best.pt --num-images 16 --sampling-steps 100
```

Para Colab:

```python
!python generate.py --checkpoint checkpoints/best.pt --num-images 16 --sampling-steps 100
```

Los resultados se guardan en `results/anime_faces_grid.png` y `results/images/`. Puedes usar `--num-images 32` para generar más rostros. `--sampling-steps 100` usa DDIM para reducir el coste de generación frente a ejecutar los 1000 pasos completos.

## 8. Qué hace cada archivo

- `check_dataset.py`: verifica la carpeta del dataset y crea una vista previa para revisar las imágenes.
- `dataset.py`: localiza las imágenes, las convierte a RGB, cambia su tamaño a 64x64, aplica volteo horizontal aleatorio y normaliza los píxeles a [-1, 1].
- `unet.py`: define la U-Net con bloques residuales, embeddings temporales y atención espacial.
- `diffusion.py`: define el calendario lineal de ruido, la ecuación directa y el muestreo inverso DDIM.
- `train.py`: entrena la U-Net para predecir el ruido con pérdida MSE, AdamW, precisión mixta en GPU y pesos EMA.
- `generate.py`: carga los pesos EMA y crea imágenes nuevas desde ruido aleatorio.

## 9. Parámetros iniciales del proyecto

| Parámetro | Valor inicial |
|---|---:|
| Resolución | 64 x 64 RGB |
| Pasos de difusión para entrenamiento | 1000 |
| Beta inicial / final | 0.0001 / 0.02 |
| Batch size | 32 |
| Épocas | 20 |
| Learning rate | 0.0002 |
| Optimizador | AdamW |
| Función de pérdida | MSE entre ruido real y predicho |
| Arquitectura | U-Net con bloques residuales y atención |
| Generación | DDIM, 100 pasos por defecto |

Son valores iniciales para un experimento académico; deben reportarse los valores realmente usados, la GPU y el tiempo real de entrenamiento.

## 10. Evidencias recomendadas para la entrega

1. Captura del dataset y de la cantidad de imágenes detectadas.
2. Captura del entrenamiento donde se vea época y pérdida MSE.
3. Archivo `training_log.csv`.
4. Capturas de `samples/` y `results/anime_faces_grid.png`.
5. Comparación de imágenes reales del dataset y nuevas imágenes generadas.
6. Explicación de los ajustes y limitaciones observadas.

## Referencia del dataset

Churchill, S. & Chao, B. (2019). *Anime Face Dataset*. Kaggle. <https://www.kaggle.com/datasets/splcher/animefacedataset>. El dataset también se describe como un conjunto de decenas de miles de rostros de anime; la cantidad efectiva debe verificarse al descargar y contar los archivos.

**Nota académica:** las imágenes de `results/` deben generarse ejecutando este proyecto con el dataset. No se incluyen resultados inventados ni pesos preentrenados.
