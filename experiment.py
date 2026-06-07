import json
import csv
import random
import time
import math
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import tensorflow_hub as hub
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Configuraciones
SEED = 42

# Diseño factorial 2x3 (algoritmo x resolución) en bloques (indoor/outdoor).
# Cada imagen se usa UNA sola vez (una observación por imagen) y no se repite
# entre grupos: indoor y outdoor son conjuntos disjuntos.
#
#   imágenes totales = observaciones totales = N_IMAGES_PER_GROUP x 2 bloques
#
# Para 1200 imágenes = 1200 observaciones, balanceadas en cada celda:
#   N_IMAGES_PER_GROUP = 600  -> 600 indoor + 600 outdoor = 1200 imágenes
#   6 tratamientos (2 alg x 3 res) x 2 bloques = 12 celdas
#   1200 / 12 = 100 imágenes por celda (100 observaciones por celda)
N_IMAGES_PER_GROUP = 600
ALGORITHMS = ['blazepose', 'movenet']
RESOLUTIONS = ['high', 'medium', 'low']

def get_resolution_size(res):
    if res == 'high':
        return None
    elif res == 'medium':
        return (256, 256)
    elif res == 'low':
        return (128, 128)
    return None

def main():
    # 1. Cargar anotaciones GT
    print("Cargando anotaciones GT...")
    gt_file = 'Archive/annotations/person_keypoints_val2017.json'
    if not os.path.exists(gt_file):
        print(f"Error: No se encontró {gt_file}")
        print("Ejecuta: python download_annotations.py")
        return

    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
        
    gt_dict = {}
    for ann in gt_data['annotations']:
        if ann.get('category_id') != 1:  # Ignorar si no es persona
            continue
            
        img_id = ann['image_id']
        keypoints = ann['keypoints']  # Formato: [x, y, v, x, y, v, ...]
        
        # Calcular cuántos keypoints son visibles (v == 2)
        num_visible = sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] == 2)
        
        # Quedarse con la persona con más keypoints visibles en la imagen
        if img_id not in gt_dict or num_visible > gt_dict.get(img_id, {}).get('num_visible', -1):
            gt_dict[img_id] = {
                'keypoints': keypoints,
                'bbox': ann['bbox'],  # [x, y, w, h]
                'num_visible': num_visible
            }
            
    # 2. Seleccionar imágenes
    print("Seleccionando imágenes...")
    
    with open('data/images_indoor.json', 'r') as f:
        indoor_data = json.load(f)
    with open('data/images_outdoor.json', 'r') as f:
        outdoor_data = json.load(f)
        
    # Filtrar imágenes que tengan GT válido
    indoor_valid = [img for img in indoor_data if img['image_id'] in gt_dict]
    outdoor_valid = [img for img in outdoor_data if img['image_id'] in gt_dict]
    
    # Aleatorizar listas
    rnd_sel = random.Random(SEED)
    rnd_sel.shuffle(indoor_valid)
    rnd_sel.shuffle(outdoor_valid)

    # Número de tratamientos (combinaciones algoritmo x resolución)
    n_treatments = len(ALGORITHMS) * len(RESOLUTIONS)  # 6

    # Garantizar que ambos grupos tengan EXACTAMENTE la misma cantidad de
    # imágenes y que ese número sea divisible entre los tratamientos, para que
    # cada celda (bloque x tratamiento) quede perfectamente balanceada. Si algún
    # bloque tiene menos imágenes válidas, se recorta hacia abajo manteniendo el
    # balance.
    n_per_group = min(N_IMAGES_PER_GROUP, len(indoor_valid), len(outdoor_valid))
    n_per_group = (n_per_group // n_treatments) * n_treatments  # múltiplo de 6
    per_cell = n_per_group // n_treatments  # imágenes por celda y por bloque
    if n_per_group < N_IMAGES_PER_GROUP:
        print(f"AVISO: se pidieron {N_IMAGES_PER_GROUP} por grupo, pero solo hay "
              f"{len(indoor_valid)} indoor y {len(outdoor_valid)} outdoor válidas.")
        print(f"       Se usarán {n_per_group} por grupo ({per_cell} por celda, "
              f"{n_per_group * 2} imágenes/observaciones en total).")

    # Tomar las primeras n_per_group de cada bloque
    indoor_selected = indoor_valid[:n_per_group]
    for img in indoor_selected:
        img['block'] = 'indoor'
        img['path'] = os.path.join('Archive', 'images', 'indoor', img['file_name'])
        
    outdoor_selected = outdoor_valid[:n_per_group]
    for img in outdoor_selected:
        img['block'] = 'outdoor'
        img['path'] = os.path.join('Archive', 'images', 'outdoor', img['file_name'])
        
    # Unir bloques (cada uno ya está acotado a n_per_group imágenes únicas)
    
    # 3. Construir la lista de corridas (UNA observación por imagen)
    print("Construyendo lista de corridas...")

    # Las 6 combinaciones de tratamiento (algoritmo x resolución)
    treatments = [(alg, res) for alg in ALGORITHMS for res in RESOLUTIONS]

    rnd_run = random.Random(SEED + 1)
    runs = []

    # Para cada bloque, asignar tratamientos de forma balanceada: 'per_cell'
    # imágenes a cada uno de los 6 tratamientos. Cada imagen recibe exactamente
    # un tratamiento, por lo que no se repite entre corridas ni entre grupos.
    for block_images in (indoor_selected, outdoor_selected):
        assignment = []
        for t in treatments:
            assignment.extend([t] * per_cell)  # per_cell veces cada tratamiento
        rnd_run.shuffle(assignment)            # mezclar qué imagen recibe qué tratamiento

        for img, (alg, res) in zip(block_images, assignment):
            runs.append({
                'image': img,
                'algorithm': alg,
                'resolution': res
            })

    # Aleatorizar el orden de ejecución del experimento
    rnd_run.shuffle(runs)
    
    # Cargar modelos
    print("Cargando modelo BlazePose...")
    model_path = 'pose_landmarker_heavy.task'
    if not os.path.exists(model_path):
        print(f"No se encontró el modelo '{model_path}'. Descargándolo desde MediaPipe...")
        import urllib.request
        import sys
        model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        try:
            def progress_hook(count, block_size, total_size):
                pct = min(count * block_size / total_size * 100, 100)
                sys.stdout.write(f"\rProgreso: {pct:.1f}%")
                sys.stdout.flush()
                if pct >= 100:
                    print()
            urllib.request.urlretrieve(model_url, model_path, reporthook=progress_hook)
            print("Modelo descargado con éxito.")
        except Exception as e:
            print(f"Error al descargar el modelo: {e}")
            return

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    pose_estimator = vision.PoseLandmarker.create_from_options(options)
    
    print("Cargando modelo MoveNet...")
    # Para evitar logs excesivos de TF:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    movenet_model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    movenet_infer = movenet_model.signatures['serving_default']
    
    # Mapeo MediaPipe (33 kp) -> COCO (17 kp)
    mp_to_coco = {
        0: 0,   # nose
        2: 1,   # left_eye
        5: 2,   # right_eye
        7: 3,   # left_ear
        8: 4,   # right_ear
        11: 5,  # left_shoulder
        12: 6,  # right_shoulder
        13: 7,  # left_elbow
        14: 8,  # right_elbow
        15: 9,  # left_wrist
        16: 10, # right_wrist
        23: 11, # left_hip
        24: 12, # right_hip
        25: 13, # left_knee
        26: 14, # right_knee
        27: 15, # left_ankle
        28: 16  # right_ankle
    }
    
    results_summary = {alg: {res: [] for res in RESOLUTIONS} for alg in ALGORITHMS}

    # Filas que se volcarán al CSV (una por corrida)
    csv_rows = []

    print(f"Total de observaciones: {len(runs)}")
    print("Iniciando experimento...\n")
    
    # Ejecutar corridas
    for idx, run in enumerate(runs):
        # Orden de corrida (1-based) según el orden aleatorizado del experimento
        orden_corrida = idx + 1

        img_info = run['image']
        alg = run['algorithm']
        res = run['resolution']
        img_path = img_info['path']
        
        # Cargar imagen
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[{orden_corrida:4d}] Error: No se pudo cargar imagen {img_path}")
            continue
            
        h_orig, w_orig = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 4 & 5 (Pre-proceso): Aplicar resize según resolución para degradar info
        target_size = get_resolution_size(res)
        if target_size:
            img_process = cv2.resize(img_rgb, target_size)
        else:
            img_process = img_rgb.copy()
            
        start_time = time.time()
        
        pred_keypoints = [None] * 17
        
        if alg == 'blazepose':
            # 4. Inferencia — BlazePose
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_process)
            results = pose_estimator.detect(mp_image)
            
            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                landmarks = results.pose_landmarks[0]
                for mp_idx, coco_idx in mp_to_coco.items():
                    lm = landmarks[mp_idx]
                    # Las coordenadas están normalizadas [0, 1] respecto a img_process, 
                    # lo cual equivale a la normalización de la imagen original.
                    x_pix = lm.x * w_orig
                    y_pix = lm.y * h_orig
                    pred_keypoints[coco_idx] = (x_pix, y_pix)
                    
        elif alg == 'movenet':
            # 5. Inferencia — MoveNet Thunder
            # MoveNet espera explícitamente tensor de 256x256 int32
            img_movenet = cv2.resize(img_process, (256, 256))
            input_tensor = tf.expand_dims(img_movenet, axis=0)
            input_tensor = tf.cast(input_tensor, dtype=tf.int32)
            
            outputs = movenet_infer(input_tensor)
            # El shape es [1, 1, 17, 3] -> [y_norm, x_norm, score]
            keypoints_with_scores = outputs['output_0'].numpy()[0, 0, :, :]
            
            for i in range(17):
                y_norm, x_norm, score = keypoints_with_scores[i]
                if score >= 0.2:
                    x_pix = x_norm * w_orig
                    y_pix = y_norm * h_orig
                    pred_keypoints[i] = (x_pix, y_pix)
                    
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        
        # 6. Calcular PCK@0.5
        gt_info = gt_dict[img_info['image_id']]
        gt_kps = gt_info['keypoints']
        bbox = gt_info['bbox']
        _, _, bw, bh = bbox
        
        dist_threshold = 0.5 * math.sqrt(bw**2 + bh**2)
        
        correct = 0
        visible_gt = 0
        detected = 0
        
        for i in range(17):
            gx = gt_kps[i*3]
            gy = gt_kps[i*3 + 1]
            gv = gt_kps[i*3 + 2]
            
            if gv == 2:  # Solo evaluar keypoints con visibility == 2
                visible_gt += 1
                pred = pred_keypoints[i]
                if pred is not None:
                    px, py = pred
                    dist = math.sqrt((gx - px)**2 + (gy - py)**2)
                    if dist <= dist_threshold:
                        correct += 1
            
            if pred_keypoints[i] is not None:
                detected += 1
                
        if visible_gt > 0:
            pck = (correct / visible_gt) * 100
        else:
            pck = 0.0
            
        results_summary[alg][res].append(pck)

        # Acumular fila para el CSV (incluye el orden de corrida)
        csv_rows.append({
            'orden_corrida': orden_corrida,
            'image_id': img_info['image_id'],
            'file_name': img_info['file_name'],
            'block': img_info['block'],
            'algorithm': alg,
            'resolution': res,
            'pck': round(pck, 2),
            'kp_vis': visible_gt,
            'kp_det': detected,
            'tiempo_ms': round(elapsed_ms, 1),
        })

        # 7. Imprimir resultados por consola
        # Formato: [run_order] image_id | bloque | algoritmo | resolucion | PCK=XX.X% | kp_vis=N | kp_det=M | tiempo=Xms
        print(f"[{orden_corrida:4d}] {img_info['image_id']:10d} | {img_info['block']:7s} | {alg:10s} | {res:6s} | PCK={pck:5.1f}% | kp_vis={visible_gt:2d} | kp_det={detected:2d} | tiempo={elapsed_ms:4.0f}ms")

    # Guardar resultados en CSV
    csv_path = 'resultados.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['orden_corrida', 'image_id', 'file_name', 'block', 'algorithm', 'resolution', 'pck', 'kp_vis', 'kp_det', 'tiempo_ms']
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV guardado en: {csv_path}  ({len(csv_rows)} filas)")

    # Imprimir resumen final
    print("\n=== RESUMEN ===")
    for alg in ALGORITHMS:
        for res in RESOLUTIONS:
            pcks = results_summary[alg][res]
            if len(pcks) > 0:
                mean_pck = np.mean(pcks)
                sd_pck = np.std(pcks, ddof=1) if len(pcks) > 1 else 0.0
                n = len(pcks)
                print(f"{alg:10s} {res:7s}: mean={mean_pck:5.1f}  sd={sd_pck:5.1f}  n={n}")
            else:
                print(f"{alg:10s} {res:7s}: n=0")

if __name__ == '__main__':
    main()