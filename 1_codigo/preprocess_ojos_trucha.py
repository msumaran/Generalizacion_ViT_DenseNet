"""
Preprocesamiento de imagenes de ojos de trucha.

Homologa las fotos de cabeza completa al formato del DNC:
  - Detecta el OJO del pez (ver algoritmo abajo)
  - Recorta una region cuadrada centrada en el ojo, con contexto de cabeza/piel
    alrededor (calibrado visualmente contra Mendeley/DNC, EYE_PAD=2.5)
  - Normaliza brillo/contraste (CLAHE) para igualar la calidad DNC
  - Redimensiona a TARGET_SIZE x TARGET_SIZE
  - Guarda en ojos_procesados/<dia>/ y un CSV con el score de cada deteccion

ALGORITMO DE DETECCION (3 etapas)
---------------------------------
1) PROPUESTA - Fast Radial Symmetry Transform (Loy & Zelinsky 2003) con
   polaridad oscura y multi-radio. Acumula votos de gradiente hacia el centro:
   solo una estructura con soporte radial en 360 grados (la pupila) genera un
   pico fuerte. Es mucho mas robusto que Hough/SimpleBlobDetector frente a los
   brillos especulares y manchas redondeadas de la piel de trucha mojada.

2) VERIFICACION - para cada uno de los 60 candidatos se prueba un rango de
   radios de iris y se extraen 15 features del patron anular: pupila oscura y
   UNIFORME, iris mas claro / mas calido (LAB a,b) / mas saturado que la piel
   de alrededor, consistente en los 12 sectores angulares, y sin fondo blanco
   (tecnopor) en la corona exterior -- esto ultimo descarta la punta del hocico,
   que es el falso positivo mas frecuente.

3) RANKING - combinacion lineal aprendida por regresion logistica L2 sobre 54
   imagenes etiquetadas a mano (leave-one-out CV: 52/54 = 96% top-1).

PRECISION MEDIDA (inspeccion visual, no solo conteo de "detectadas")
--------------------------------------------------------------------
  - Muestra fresca de 30 imagenes NO usadas para entrenar:  27/30 = 90%
  - Auditoria de 40 recortes al azar del bucket 'ok':       38/40 = 95%
  - Auditoria de 16 recortes al azar del bucket 'dudosa':    4/16 = 25%
  => sobre el dataset completo, ~88% de los recortes contienen el ojo real.

Filtrar por estado == 'ok' en deteccion_scores.csv sube la precision a ~95%
a costa de descartar ~11% de las imagenes (de las cuales 1 de cada 4 era
en realidad correcta).

Uso:
    python preprocess_ojos_trucha.py              # procesa todo
    python preprocess_ojos_trucha.py --preview    # + grid de detecciones
    python preprocess_ojos_trucha.py --single X   # debug de una imagen
"""

import os
import argparse
import multiprocessing as mp
import time
import csv

import cv2
import numpy as np
from PIL import Image, ImageDraw

OJOS_DIR    = "Trucha"
OUT_DIR     = "ojos_procesados"
TARGET_SIZE = 330          # promedio DNC (331x340)
EYE_PAD     = 2.5          # calibrado visualmente contra Mendeley/DNC:
                           # el radio detectado "r" es solo el radio del
                           # IRIS (no incluye cabeza/piel circundante), asi
                           # que con pad=0.43 el recorte quedaba demasiado
                           # ajustado (el ojo llenaba ~90-100% del frame)
                           # frente al ~30-50% tipico en las fotos de
                           # Mendeley. pad=2.5 deja suficiente contexto de
                           # cabeza/piel visible para igualar esa composicion
                           # (ver comparacion visual: pad_final_calibration.png)

# Por debajo de este score la deteccion se marca como dudosa en el CSV.
# (En la muestra de validacion separaba 27 aciertos de 3 fallos sin error.)
DETECT_MIN_SCORE = 0.5

DIAS = ["dia-1", "dia-3", "dia-5", "dia-7", "dia-9"]

# Etiquetas para las subcarpetas de salida
DIA_LABEL = {
    "dia-1": "highly_fresh",
    "dia-3": "highly_fresh",
    "dia-5": "fresh",
    "dia-7": "not_fresh",
    "dia-9": "not_fresh",
}

DET_SHORT = 420        # lado corto de la imagen para la deteccion
N_SECT = 12            # sectores angulares del anillo del iris
FRST_RADII = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]
R_LIST = list(range(7, 30))
NCAND = 60

# Ranker lineal aprendido (regresion logistica L2, 54 imagenes etiquetadas).
# Features: [minsec, meanc, warm, ringout, seccv, sym, dark, corestd,
#            ringstd, catch, satring, ringabs, bgring, bgout, outstd]
_W = np.array([0.01089, -0.34411, 0.64001, 1.27509, 0.09151, 1.08903, 0.25335, -0.67266, -0.24096, -0.36672, 0.06535, -0.37655, -0.11604, -0.14779, -0.30123], np.float64)
_B = -4.59888
_MU = np.array([13.6346, 37.0955, 0.3857, -1.7529, 0.1907, 0.2126, 104.2578, 15.8810, 26.2535, 41.5709, 36.9766, 102.8377, 0.0775, 0.0916, 33.0635], np.float64)
_SD = np.array([11.5198, 16.0357, 2.6665, 15.0844, 0.1386, 0.1807, 37.8743, 8.5513, 11.7777, 25.5136, 24.9356, 40.1363, 0.1502, 0.1484, 12.2269], np.float64)

_MC = {}


def _masks(r):
    if r in _MC:
        return _MC[r]
    R = int(np.ceil(r * 1.7))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    d2 = (xx ** 2 + yy ** 2).astype(np.float32)
    ang = (np.arctan2(yy, xx) + np.pi) / (2 * np.pi)
    sect = np.minimum((ang * N_SECT).astype(np.int32), N_SECT - 1)
    core = d2 <= (r * 0.45) ** 2
    ring = (d2 > (r * 0.62) ** 2) & (d2 <= (r * 1.02) ** 2)
    out = (d2 > (r * 1.25) ** 2) & (d2 <= (r * 1.70) ** 2)
    sm = [ring & (sect == k) for k in range(N_SECT)]
    _MC[r] = (R, core, ring, out, sm)
    return _MC[r]


def _prep(img_bgr):
    h, w = img_bgr.shape[:2]
    scale = DET_SHORT / min(h, w)
    small = cv2.resize(img_bgr, (int(round(w * scale)), int(round(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return small, scale


def frst(gray, radii, alpha=2.0, grad_pct=70):
    """Fast Radial Symmetry Transform, polaridad oscura (centros oscuros)."""
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    thr = np.percentile(mag, grad_pct)
    ys, xs = np.nonzero(mag > max(thr, 1e-3))
    if ys.size == 0:
        return np.zeros_like(g)
    m = mag[ys, xs]
    ux, uy = gx[ys, xs] / m, gy[ys, xs] / m
    H, W = g.shape
    S = np.zeros((H, W), np.float32)
    for n in radii:
        kn = 9.9
        nx = np.clip((xs - np.round(ux * n)).astype(np.int32), 0, W - 1)
        ny = np.clip((ys - np.round(uy * n)).astype(np.int32), 0, H - 1)
        flat = ny * W + nx
        O = np.zeros(H * W, np.float32)
        M = np.zeros(H * W, np.float32)
        np.add.at(O, flat, 1.0)
        np.add.at(M, flat, m)
        O = np.minimum(O, kn).reshape(H, W)
        F = (O / kn) ** alpha * (M.reshape(H, W) / kn)
        k = int(2 * round(0.25 * n) + 1)
        S += cv2.GaussianBlur(F, (max(k, 3), max(k, 3)), 0.25 * n)
    return S / len(radii)


def detect_eye(img_bgr):
    """
    Devuelve (cx, cy, r_iris, score) en coordenadas ORIGINALES, o None.
    score >= DETECT_MIN_SCORE => deteccion confiable; por debajo => dudosa.
    """
    small, scale = _prep(img_bgr)
    H, W = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    warm = (np.maximum(lab[:, :, 1].astype(np.float32) - 128, 0) +
            np.maximum(lab[:, :, 2].astype(np.float32) - 128, 0))
    # mascara de fondo (tecnopor / plastico blanco): claro y desaturado
    bg = ((val > 185) & (sat < 45)).astype(np.float32)
    g = gray.astype(np.float32)

    S = frst(cv2.GaussianBlur(gray, (5, 5), 1.2), FRST_RADII)
    S = cv2.GaussianBlur(S, (7, 7), 2.0)
    # la pupila es oscura: descartar picos sobre zonas claras
    S = np.where(cv2.GaussianBlur(g, (11, 11), 3.0) < 170, S, 0)

    mx = cv2.dilate(S, np.ones((13, 13), np.uint8))
    pk = (S == mx) & (S > S.max() * 0.04)
    ys, xs = np.nonzero(pk)
    if ys.size == 0:
        return None
    o = np.argsort(-S[ys, xs])[:NCAND]
    ys, xs = ys[o], xs[o]
    smax = float(S.max()) + 1e-6

    feats, meta = [], []
    for cy, cx in zip(ys, xs):
        cy, cx = int(cy), int(cx)
        sym = float(S[cy, cx]) / smax
        best = None
        for r in R_LIST:
            R, core, ring, out, sm = _masks(r)
            if cx - R < 0 or cy - R < 0 or cx + R >= W or cy + R >= H:
                continue
            sl = (slice(cy - R, cy + R + 1), slice(cx - R, cx + R + 1))
            gp = g[sl]
            v_core = float(gp[core].mean())
            secs = np.array([gp[m].mean() for m in sm], np.float32)
            minsec = float(secs.min() - v_core)
            if minsec <= 0:
                continue
            if best is None or minsec > best[0]:
                v_ring = float(gp[ring].mean())
                v_out = float(gp[out].mean())
                wp, sp, bp = warm[sl], sat[sl], bg[sl]
                best = (minsec, r, [
                    minsec,
                    v_ring - v_core,
                    float(wp[ring].mean() - wp[out].mean()),
                    v_ring - v_out,
                    float(secs.std() / (secs.mean() + 1e-6)),
                    sym,
                    170.0 - v_core,
                    float(gp[core].std()),
                    float(gp[ring].std()),
                    float(np.percentile(gp[core], 98) - np.percentile(gp[core], 40)),
                    float(sp[ring].mean()),
                    v_ring,
                    float(bp[ring].mean()),
                    float(bp[out].mean()),
                    float(gp[out].std())])
        if best is None:
            continue
        feats.append(best[2])
        meta.append((cx, cy, best[1]))

    if not feats:
        return None
    Z = (np.asarray(feats, np.float64) - _MU) / _SD
    sc = Z @ _W + _B
    i = int(np.argmax(sc))
    cx, cy, r = meta[i]
    return (int(round(cx / scale)), int(round(cy / scale)),
            int(round(r / scale)), float(sc[i]))



def crop_eye(img_bgr, cx, cy, r, pad=EYE_PAD, size=TARGET_SIZE):
    """Recorta un cuadrado centrado en el ojo y lo homologa al formato DNC."""
    h, w = img_bgr.shape[:2]
    half = max(int(r * (1 + pad)), 8)
    x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
    # si el ojo esta cerca del borde, rellenar por replicacion (no descentrar)
    px1, py1 = max(0, -x1), max(0, -y1)
    px2, py2 = max(0, x2 - w), max(0, y2 - h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    if px1 or py1 or px2 or py2:
        crop = cv2.copyMakeBorder(crop, py1, py2, px1, px2, cv2.BORDER_REPLICATE)

    # CLAHE sobre L para homologar contraste con DNC
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    crop = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4)
    # simular calidad DNC: ligero blur (fotos macro de celular)
    return cv2.GaussianBlur(crop, (3, 3), 0.8)


# ---------------------------------------------------------------------------
def _process_one(args):
    """Worker de multiprocessing: procesa un archivo."""
    src_path, out_path = args
    img = cv2.imread(src_path)
    if img is None:
        return 'error', 0.0, None

    eye = detect_eye(img)
    if eye is None:
        # sin ningun candidato: recorte central del tercio medio (ultimo recurso)
        h, w = img.shape[:2]
        s = min(h, w) // 2
        crop = img[h // 4:h // 4 + s, w // 4:w // 4 + s]
        crop = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE))
        cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return 'fallback', 0.0, crop

    cx, cy, r, score = eye
    crop = crop_eye(img, cx, cy, r)
    if crop is None:
        return 'error', float(score), None

    cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return ('ok' if score >= DETECT_MIN_SCORE else 'dudosa'), float(score), crop


def process_all(preview=False, workers=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    if workers is None:
        workers = max(1, mp.cpu_count() - 1)

    tot = {'ok': 0, 'dudosa': 0, 'fallback': 0, 'error': 0}
    rows = []
    preview_imgs = []
    t0 = time.time()

    for dia in DIAS:
        src_dir = os.path.join(OJOS_DIR, dia)
        if not os.path.isdir(src_dir):
            print(f"[SKIP] {src_dir} no existe")
            continue
        out_sub = os.path.join(OUT_DIR, dia)
        os.makedirs(out_sub, exist_ok=True)

        files = sorted(f for f in os.listdir(src_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        job_args = [(os.path.join(src_dir, f), os.path.join(out_sub, f))
                    for f in files]

        cnt = {'ok': 0, 'dudosa': 0, 'fallback': 0, 'error': 0}
        n_prev = 0          # hasta 6 muestras de preview por dia
        with mp.Pool(workers) as pool:
            for fname, (status, score, crop) in zip(
                    files, pool.imap(_process_one, job_args, chunksize=4)):
                cnt[status] += 1
                tot[status] += 1
                rows.append((dia, fname, f"{score:.3f}", status))
                if status != 'ok':
                    print(f"  [{status.upper():8s}] {dia}/{fname}  score={score:.2f}")
                if preview and crop is not None and n_prev < 6:
                    preview_imgs.append((status, dia, fname, crop))
                    n_prev += 1

        print(f"[{dia}]  ok={cnt['ok']}  dudosa={cnt['dudosa']}  "
              f"fallback={cnt['fallback']}  error={cnt['error']}  -> {out_sub}/")

    elapsed = time.time() - t0
    n = sum(tot.values())
    print(f"\nTOTAL: {n} imagenes  |  ok={tot['ok']}  dudosa={tot['dudosa']}  "
          f"fallback={tot['fallback']}  error={tot['error']}  "
          f"({elapsed:.1f}s, {workers} workers)")

    csv_path = os.path.join(OUT_DIR, "deteccion_scores.csv")
    with open(csv_path, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["dia", "archivo", "score", "estado"])
        wtr.writerows(rows)
    print(f"Scores por imagen: {csv_path}")
    print(f"  (estado='dudosa' => score < {DETECT_MIN_SCORE}; revisar o excluir)")

    if preview and preview_imgs:
        _save_preview(preview_imgs)


def _save_preview(items, cols=6, cell=220):
    rows_n = (len(items) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell + (cols - 1) * 4,
                             rows_n * (cell + 30) + (rows_n - 1) * 4), (230, 230, 230))
    draw = ImageDraw.Draw(grid)
    for i, (status, dia, fname, crop_bgr) in enumerate(items):
        pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        pil = pil.resize((cell, cell), Image.LANCZOS)
        col, row = i % cols, i // cols
        x, y = col * (cell + 4), row * (cell + 30) + 4
        grid.paste(pil, (x, y + 26))
        color = (20, 160, 100) if status == 'ok' else (200, 60, 50)
        draw.rectangle([x, y, x + cell, y + 24], fill=color)
        draw.text((x + cell // 2, y + 12), f"{dia} [{status}]",
                  fill=(255, 255, 255), anchor="mm")
    out = os.path.join(OUT_DIR, "preview_deteccion.jpg")
    grid.save(out, quality=90)
    print(f"Preview guardado: {out}")


def preview_single(path):
    """Corre el detector sobre una imagen individual, guarda la
    visualizacion + el recorte, y da un veredicto ok/dudosa (debug/uso
    manual para fotos nuevas que todavia no pasaron por el pipeline)."""
    img = cv2.imread(path)
    if img is None:
        print("No se pudo abrir:", path)
        return
    eye = detect_eye(img)
    if not eye:
        print("SIN DETECCION: no se encontro ningun candidato a ojo en la imagen.")
        print("Verdicto: MAL -- revisar la foto (ojo no visible, muy oscura, o fuera de rango de radios).")
        return
    cx, cy, r, score = eye
    ok = score >= DETECT_MIN_SCORE
    veredicto = "BIEN" if ok else "MAL"
    print(f"ojo en ({cx},{cy}) r={r} score={score:.2f}  (umbral ok: >= {DETECT_MIN_SCORE})")
    print(f"Verdicto: {veredicto} -- {'el recorte deberia mostrar el ojo centrado.' if ok else 'score bajo, revisar visualmente antes de usar este recorte.'}")
    vis = img.copy()
    color = (0, 200, 80) if ok else (0, 60, 220)
    cv2.circle(vis, (cx, cy), r, color, max(3, img.shape[1] // 500))
    out = os.path.splitext(os.path.basename(path))[0]
    cv2.imwrite(f"{out}_deteccion.jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(f"{out}_recorte.jpg", crop_eye(img, cx, cy, r))
    print(f"Guardado: {out}_deteccion.jpg (circulo verde=bien/rojo=mal) / {out}_recorte.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true",
                        help="Guarda grid de previsualizaciones")
    parser.add_argument("--single", type=str, default=None,
                        help="Procesar una sola imagen (debug)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Numero de procesos paralelos (default: CPU-1)")
    args = parser.parse_args()

    if args.single:
        preview_single(args.single)
    else:
        process_all(preview=args.preview, workers=args.workers)
