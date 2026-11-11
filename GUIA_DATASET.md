# Guía rápida: descargar y colocar el dataset

## Método 1: descargar desde Kaggle y descomprimir (Windows)

1. Abre <https://www.kaggle.com/datasets/splcher/animefacedataset> e inicia sesión.
2. Pulsa **Download**. Se descargará un archivo ZIP.
3. Dentro de la carpeta `Anime_Diffusion_Project`, crea `data` y luego `animefacedataset`.
4. Haz clic derecho en el ZIP > **Extraer todo...** y elige `Anime_Diffusion_Project/data/animefacedataset` como destino.
5. Comprueba que haya imágenes `.jpg` o `.png`. Puede quedar una estructura como cualquiera de estas:

```text
Anime_Diffusion_Project/
└── data/
    └── animefacedataset/
        ├── images/
        │   ├── 000001.jpg
        │   ├── 000002.jpg
        │   └── ...
        └── ...
```

O las imágenes pueden estar directamente dentro de `animefacedataset`. Ambas formas funcionan porque el cargador busca también en subcarpetas.

6. Abre PowerShell en la carpeta `Anime_Diffusion_Project` y ejecuta el verificador:

```powershell
python check_dataset.py --data-dir data/animefacedataset
```

Se generará `dataset_preview.png`. Revisa que se muestren rostros y que se haya encontrado una cantidad razonable de imágenes. Luego inicia el entrenamiento:

```powershell
python train.py --data-dir data/animefacedataset --epochs 20 --batch-size 32
```

## Método 2: Google Colab + Kaggle API (recomendado si no tienes GPU)

1. Abre Google Colab y cambia el entorno a GPU: **Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU**.
2. En Kaggle abre tu perfil > **Settings > API > Create New Token**. Se descargará `kaggle.json`.
3. Sube los archivos de este proyecto a Colab, por ejemplo a `/content/Anime_Diffusion_Project`.
4. Ejecuta estas celdas:

```python
%cd /content/Anime_Diffusion_Project
!pip install -r requirements.txt
```

```python
!pip install -q kaggle
from google.colab import files
uploaded = files.upload()  # Selecciona el archivo kaggle.json
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!mkdir -p data/animefacedataset
!kaggle datasets download -d splcher/animefacedataset -p data/animefacedataset --unzip
```

5. Comprueba que se detectan imágenes y guarda una vista previa:

```python
!python check_dataset.py --data-dir data/animefacedataset
```

6. Si la vista previa se ve correcta, ejecuta el entrenamiento final (en Colab el tiempo dependerá de la GPU asignada):

```python
!python train.py --data-dir data/animefacedataset --epochs 20 --batch-size 32 --base-channels 64 --num-workers 2
```

7. Para generar rostros al terminar:

```python
!python generate.py --checkpoint checkpoints/best.pt --num-images 16 --sampling-steps 100
```

**Seguridad:** no compartas `kaggle.json`, no lo subas a GitHub y no lo incluyas en la entrega. Contiene la clave de acceso a tu cuenta de Kaggle.

## Si aparece un error de memoria CUDA

Reduce primero el batch size:

```bash
python train.py --data-dir data/animefacedataset --epochs 20 --batch-size 8 --base-channels 32
```

Si el error continúa, usa una GPU con más memoria. CPU funciona, pero puede ser demasiado lento para entrenar el dataset completo.
