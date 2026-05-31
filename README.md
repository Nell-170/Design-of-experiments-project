# Diseño de Experimentos — Proyecto Grupal

Experimento factorial **2×3** que compara dos algoritmos de estimación de pose humana sobre imágenes estáticas del dataset **COCO val2017**.

| Factor | Niveles |
|--------|---------|
| **A — Algoritmo** | MediaPipe BlazePose Full · MoveNet Thunder v4 |
| **B — Resolución de entrada** | Alta (original) · Media (256×256) · Baja (128×128) |

**Variable de respuesta:** PCK@0.5 (Percentage of Correct Keypoints, umbral = 50 % de la diagonal del bounding box; solo keypoints con visibilidad COCO = 2).  
**Bloques:** tipo de iluminación — interior (`indoor`) / exterior (`outdoor`).  
**Semilla de aleatorización:** `seed = 42`, 300 imágenes en total (50 por celda del factorial).

---

## Estructura del repositorio

```
.
├── Archive/                        # Datos locales — gitignoreados
│   ├── annotations/                # Anotaciones COCO (instances + keypoints)
│   └── images/
│       ├── indoor/                 # Imágenes descargadas — bloque interior
│       └── outdoor/                # Imágenes descargadas — bloque exterior
├── data/                           # Salidas del split script
│   ├── images_indoor.json          # image_id clasificados como indoor
│   ├── images_outdoor.json         # image_id clasificados como outdoor
│   ├── images_ambiguous.json       # Imágenes no clasificables (excluidas)
│   └── blocks_summary.json         # Resumen del split indoor/outdoor
├── docs/                           # Entregables y enunciado del curso
│   ├── propuesta_final.docx
│   └── proyecto_enunciado.pdf
├── scripts/                        # Scripts de setup y utilidades
│   ├── download_annotations.py     # Descarga anotaciones COCO val2017
│   ├── download_and_organize.py    # Descarga y organiza las imágenes en carpetas
│   └── split_indoor_outdoor_blocks.py  # Clasifica imágenes indoor/outdoor
├── experiment.py                   # Runner del experimento — genera observaciones PCK@0.5
├── dependencies.txt                # Dependencias del proyecto
└── README.md
```

---

## Configuración del entorno local

### Requisitos previos
- **Python 3.12** (`python3.12 --version`)

### 1. Clonar el repositorio

```bash
git clone https://github.com/Nell-170/Design-of-experiments-project.git
cd Design-of-experiments-project
```

### 2. Crear y activar el entorno virtual

```bash
# Crear entorno con Python 3.12
python3.12 -m venv .venv

# Activar (macOS / Linux)
source .venv/bin/activate

# Activar (Windows)
.venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r dependencies.txt
```

### 4. Obtener las anotaciones de COCO

```bash
python scripts/download_annotations.py
```

Descarga `annotations_trainval2017.zip` (~241 MB) y extrae solo los dos archivos necesarios en `Archive/annotations/`. Si ya existen, no los vuelve a descargar.

### 5. Descargar y organizar las imágenes

El script descarga **solo las 1 503 imágenes necesarias** directamente desde los servidores de COCO (no el zip completo de ~6 GB):

```bash
python scripts/download_and_organize.py
```

Esto crea `Archive/images/indoor/` y `Archive/images/outdoor/` con las imágenes clasificadas.

Opciones disponibles:
```
--output-dir  DIR    Directorio destino (default: Archive/images)
--workers     N      Hilos de descarga paralela (default: 16)
```

### 6. Ejecutar el runner del experimento

```bash
python experiment.py
```


---

## Regenerar el split indoor/outdoor

Si necesitas regenerar los archivos `images_*.json` a partir de las anotaciones COCO:

```bash
python scripts/split_indoor_outdoor_blocks.py
# Con anotaciones en rutas no default:
python scripts/split_indoor_outdoor_blocks.py \
    --instances Archive/annotations/instances_val2017.json \
    --keypoints Archive/annotations/person_keypoints_val2017.json \
    --output-dir data
```

---

### Mapeo MediaPipe → COCO (17 keypoints)

| MP índice | COCO índice | Nombre |
|-----------|-------------|--------|
| 0 | 0 | nose |
| 2 | 1 | left_eye |
| 5 | 2 | right_eye |
| 7 | 3 | left_ear |
| 8 | 4 | right_ear |
| 11 | 5 | left_shoulder |
| 12 | 6 | right_shoulder |
| 13 | 7 | left_elbow |
| 14 | 8 | right_elbow |
| 15 | 9 | left_wrist |
| 16 | 10 | right_wrist |
| 23 | 11 | left_hip |
| 24 | 12 | right_hip |
| 25 | 13 | left_knee |
| 26 | 14 | right_knee |
| 27 | 15 | left_ankle |
| 28 | 16 | right_ankle |

---

## Próximos pasos

- **Ejecutar el runner del experimento y crear el archivo CSV con los datos recolectados**.
- **Realizar el Análisis Exploratorio de Datos (EDA) en R**: (emma) 
  - Crear el script de R para calcular estadísticas descriptivas (media, varianza, desviación estándar, cuartiles, etc.).
  - Generar gráficos solicitados: boxplots (general, por factor, y combinados), histograma de la variable de respuesta y gráfico en el tiempo para verificar aleatorización.
  - Generar gráficos de posibles interacciones de los factores.
  - Realizar el análisis preliminar basado en los resultados y gráficos obtenidos.
- **Redactar el informe escrito (formato PDF)**:
  - Incluir la información de la propuesta corregida (contexto y problema).
  - Describir el diseño experimental factorial (factores, niveles, modelo de efectos de datos).
  - Definir las hipótesis nulas y alternativas para factores e interacciones.
  - Incluir el EDA y análisis preliminar.
- **Preparar la presentación oral**:
  - Elaborar el material visual (PowerPoint o equivalente).
  - Estructurar la presentación para que dure entre 8 y 10 minutos.
- **Completar evaluaciones individuales**: Llenar el documento de autoevaluación y co-evaluación en formato PDF.
---

## Integrantes

- Christopher Zúñiga
- Mario Cordero
- Emmanuel Gonzales
- Annabelle Porras 

---

## Curso

CI-0131 Diseño de Experimentos — Universidad de Costa Rica  
Entrega del Avance: **lunes 8 de junio de 2026**