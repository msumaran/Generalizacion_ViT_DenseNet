"""Carga de los 3 dominios usados en la evaluacion cross-dataset del plan de
tesis (seccion 5.3 Poblacion y muestra / seccion "Diseno experimental"):

  Mendeley Data  -> dominio FUENTE  -> carpeta local: DataSetNotClean_4k/
                    (Prasetyo et al., 2022a; 8 especies, Indonesia)
  OneDrive       -> dominio OBJETIVO 1 -> carpeta local: ojos_procesados/
                    (Mamani & Pedraza, 2023; trucha, Peru)
  Kaggle         -> dominio OBJETIVO 2 -> carpeta local: rohu_ojos_procesados/
                    (JIS College of Engineering, 2024; Rohu, India)

IMPORTANTE -- por que se usan las carpetas "*_procesados" y no las fotos
originales (Trucha/, Rohu/):
  Las fotos originales de Trucha y Rohu son fotos de cabeza completa a
  resolucion de camara (~4160x3120 y ~4032x3024 respectivamente), mientras
  que Mendeley (DataSetNotClean_4k, el dataset con el que se entrenaron los
  checkpoints) ya viene recortado al OJO (resolucion ~300x300-440x440).
  Alimentar al modelo las fotos completas sin procesar mediria una mezcla de
  "cambio de dominio" + "cambio de composicion/encuadre", no una comparacion
  valida de generalizacion.

  Por eso este proyecto usa las salidas YA GENERADAS de los scripts de
  preprocesamiento del proyecto (preprocess_ojos_trucha.py y
  preprocess_ojos_rohu.py), que:
    1) detectan el ojo del pez (Fast Radial Symmetry Transform, 3 etapas),
    2) recortan un cuadrado centrado en el ojo (ojo ~70% del frame, igual
       que en Mendeley/DNC),
    3) normalizan brillo/contraste (CLAHE),
    4) redimensionan a TARGET_SIZE=330x330 -- LA MISMA resolucion para
       Trucha y Rohu (verificado: ambas carpetas *_procesados contienen
       imagenes de exactamente 330x330 px).

  Cada deteccion tiene un score de calidad en un CSV aparte
  (ojos_procesados/deteccion_scores.csv y
  rohu_ojos_procesados/deteccion_scores_rohu.csv, columna "estado": ok/dudosa).

ESTRUCTURA DE CARPETAS -- ojos_procesados/ (Trucha) esta organizado por
CLASE (fresco/no_fresco), igual que Rohu (Fresh_Eyes/Nonfresh_Eyes) y
Mendeley (-Fresh/-Not Fresh) -- NO por dia como la carpeta original del
proyecto padre (../ojos_procesados/, que sigue dia-1..dia-9 porque otros
23 scripts del proyecto dependen de esa estructura y no se tocaron para no
romperlos). El dia queda preservado como prefijo del archivo
("dia-N_archivo.jpg"). Por eso este script SOLO funciona corriendolo desde
DENTRO de esta carpeta (cd Generalizacion_CrossDataset && python domains.py)
-- las rutas son relativas al cwd, no al proyecto padre.

  AUDITORIA VISUAL (muestra de 32 por dataset/bucket, ver
  verificacion_preprocesamiento/): la tasa real de imagenes que SI contienen
  el ojo es ~94-97% en el bucket "ok" pero solo ~35% en "dudosa" (el resto
  son escamas/aletas/fondo mal detectados). El score de verificacion NO
  separa limpiamente los "dudosa" salvables de los realmente malos (ambos
  caen en rangos de score similares), asi que filtrar por "ok" es la unica
  forma confiable de excluir el ruido sin revision manual caso por caso.

  Por eso collect_trucha() y collect_rohu_eyes() filtran por defecto
  (only_ok=True) usando estos CSV -- pasar only_ok=False para usar TODAS
  las imagenes (incluye las "dudosa", ~65% de las cuales son detecciones
  incorrectas).

  RESCATE MANUAL (3 pasadas): se revisaron visualmente las 208 imagenes
  "dudosa" -- pasadas 1-2 solo CLASIFICARON el recorte automatico ya
  generado (miniaturas 6 col., luego 5 col. mas grandes sobre lo
  descartado), encontrando 52/128 en Trucha y 31/80 en Rohu. Para las que
  seguian excluidas (76 Trucha, 49 Rohu), la pasada 3 volvio a la FOTO
  ORIGINAL sin procesar (con grilla de coordenadas superpuesta) en vez del
  recorte ya fallido -- el ojo resultaba visible casi siempre, pero
  estimar sus coordenadas a mano (sin el ajuste de radio automatico por
  imagen) solo produjo un recorte bien centrado en 30/70 Trucha y 27/49
  Rohu tras verificar cada resultado uno por uno; el resto se descarto en
  vez de forzar un recorte dudoso. Total rescatado: 82/128 Trucha, 58/80
  Rohu. Esta lista esta en verificacion_preprocesamiento/{trucha,rohu}_rescatadas.csv
  y se incluye junto con "ok" por defecto. Sigue siendo una clasificacion
  visual, no una verificacion independiente de un tercero -- revisar
  revision_manual_dudosa.csv si se quiere auditar o corregir alguna.

Todas las funciones devuelven (paths, labels) en binario:
  0 = fresco (incluye "Highly Fresh")   1 = no_fresco
"""
import csv
import os

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

MENDELEY_DIR = "DataSetNotClean_4k"
TRUCHA_DIR = "ojos_procesados"        # salida de preprocess_ojos_trucha.py (330x330)
ROHU_DIR = "rohu_ojos_procesados"     # salida de preprocess_ojos_rohu.py (330x330)

TARGET_SIZE_PREPROCESADO = 330  # resolucion comun de Trucha y Rohu ya procesados

# ojos_procesados/ dentro de este repo esta organizado por CLASE (fresco/
# no_fresco), igual que Rohu (Fresh_Eyes/Nonfresh_Eyes) y Mendeley
# (-Fresh/-Not Fresh) -- NO por dia como la carpeta original del proyecto
# padre (que sigue con dia-1..dia-9 porque otros 23 scripts del proyecto
# dependen de esa estructura; no se toco para no romperlos). El dia queda
# preservado como prefijo del nombre de archivo ("dia-N_archivo.jpg").
TRUCHA_CLASS_TO_LABEL = {"fresco": 0, "no_fresco": 1}


def _list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in EXTENSIONS)


def _load_trucha_estados(csv_path=os.path.join(TRUCHA_DIR, "deteccion_scores.csv")):
    """(clase, archivo) -> 'ok'|'dudosa', segun preprocess_ojos_trucha.py
    (el csv guarda 'dia' y 'clase'; aqui se indexa por clase+archivo para
    que coincida con la estructura fisica de carpetas fresco/no_fresco)."""
    estados = {}
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                estados[(row["clase"], row["archivo"])] = row["estado"]
    return estados


def _load_rohu_estados(csv_path=os.path.join(ROHU_DIR, "deteccion_scores_rohu.csv")):
    """(split, subset, archivo) -> 'ok'|'dudosa', segun preprocess_ojos_rohu.py."""
    estados = {}
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                estados[(row["split"], row["subset"], row["archivo"])] = row["estado"]
    return estados


_VERIF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verificacion_preprocesamiento")


def _load_trucha_rescatadas(csv_path=os.path.join(_VERIF_DIR, "trucha_rescatadas.csv")):
    """(clase, archivo) rescatados manualmente de 'dudosa' (ver docstring del modulo)."""
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path) as f:
        return {(r["clase"], r["archivo"]) for r in csv.DictReader(f)}


def _load_rohu_rescatadas(csv_path=os.path.join(_VERIF_DIR, "rohu_rescatadas.csv")):
    """(split, subset, archivo) rescatados manualmente de 'dudosa'."""
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path) as f:
        return {(r["split"], r["subset"], r["archivo"]) for r in csv.DictReader(f)}


_ROHU_FALSOS_POSITIVOS_CSV = [
    os.path.join(_VERIF_DIR, "rohu_escamas_confirmadas.csv"),  # ronda 1: 71
    os.path.join(_VERIF_DIR, "rohu_no_ojo_ronda2.csv"),        # ronda 2: 38
]


def _load_rohu_escamas_confirmadas(csv_paths=_ROHU_FALSOS_POSITIVOS_CSV):
    """(split, subset, archivo) marcadas 'ok' por el score pero confirmadas
    visualmente como falsos positivos (escama/aleta/foto borrosa sin ojo) en
    auditoria manual -- el ranker de Rohu no las distingue (ninguna tenia
    score bajo, ver docstring del modulo).

    Dos rondas (secciones 6.2 y 6.3 del informe): la ronda 1 reviso las
    1,572 imagenes 'ok' en hojas de contacto que MEZCLABAN sesiones y a esa
    densidad se le escaparon casos; la ronda 2 uso un ranking por
    "¿tiene pupila?" (detectar_no_ojos.py) para priorizar la revision. Los
    archivos tienen columnas distintas (la ronda 2 agrega rank/score), pero
    solo se leen split/subset/archivo de ambos."""
    fuera = set()
    for p in csv_paths:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            fuera |= {(r["split"], r["subset"], r["archivo"]) for r in csv.DictReader(f)}
    return fuera


def collect_mendeley(base_dir=MENDELEY_DIR):
    """Dominio FUENTE. Carpetas '<especie> - Fresh|Highly Fresh|Not Fresh'."""
    paths, labels = [], []
    for folder in sorted(os.listdir(base_dir)):
        fp = os.path.join(base_dir, folder)
        if not os.path.isdir(fp):
            continue
        cu = folder.strip().upper()
        if "NOT FRESH" in cu:
            lbl = 1
        elif "FRESH" in cu:  # cubre "Fresh" y "Highly Fresh"
            lbl = 0
        else:
            continue
        for fname in _list_images(fp):
            paths.append(os.path.join(fp, fname))
            labels.append(lbl)
    return paths, labels


def collect_trucha(base_dir=TRUCHA_DIR, only_ok=True, incluir_rescatadas=True):
    """Dominio OBJETIVO 1 (OneDrive). Carpetas 'fresco/' y 'no_fresco/' YA
    preprocesadas (ojo recortado, CLAHE, 330x330) por preprocess_ojos_trucha.py
    -- el dia original queda como prefijo del nombre de archivo.
    only_ok=True (default) excluye las detecciones marcadas 'dudosa'
    (~11% del total, ~65% de ellas son detecciones incorrectas segun
    auditoria visual -- ver docstring del modulo). incluir_rescatadas=True
    (default) vuelve a incluir las 'dudosa' que la revision manual confirmo
    que SI muestran el ojo (ver trucha_rescatadas.csv)."""
    estados = _load_trucha_estados() if only_ok else {}
    rescatadas = _load_trucha_rescatadas() if (only_ok and incluir_rescatadas) else set()
    paths, labels = [], []
    n_excluidas = n_rescatadas = 0
    for clase, lbl in TRUCHA_CLASS_TO_LABEL.items():
        fp = os.path.join(base_dir, clase)
        for fname in _list_images(fp):
            key = (clase, fname)
            if only_ok and estados.get(key) != "ok":
                if key in rescatadas:
                    n_rescatadas += 1
                else:
                    n_excluidas += 1
                    continue
            paths.append(os.path.join(fp, fname))
            labels.append(lbl)
    if only_ok:
        print(f"[collect_trucha] excluidas {n_excluidas} 'dudosa' + incluidas {n_rescatadas} rescatadas manualmente")
    return paths, labels


def collect_rohu_eyes(base_dir=ROHU_DIR, only_ok=True, incluir_rescatadas=True):
    """Dominio OBJETIVO 2 (Kaggle). Carpetas 'Training|Testing/{Fresh,Nonfresh}_Eyes'
    YA preprocesadas (ojo recortado, CLAHE, 330x330) por preprocess_ojos_rohu.py.
    Junta Training + Testing: al ser evaluacion zero-shot (sin entrenar en
    este dominio) no aplica la particion original del repositorio.
    only_ok=True (default) excluye las detecciones marcadas 'dudosa'.
    incluir_rescatadas=True (default) vuelve a incluir las 'dudosa' que la
    revision manual confirmo que SI muestran el ojo (rohu_rescatadas.csv)."""
    estados = _load_rohu_estados() if only_ok else {}
    rescatadas = _load_rohu_rescatadas() if (only_ok and incluir_rescatadas) else set()
    escamas = _load_rohu_escamas_confirmadas() if only_ok else set()
    paths, labels = [], []
    n_excluidas = n_rescatadas = n_escamas = 0
    for split in ("Training", "Testing"):
        split_dir = os.path.join(base_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for folder in sorted(os.listdir(split_dir)):
            if not folder.endswith("_Eyes"):
                continue
            fp = os.path.join(split_dir, folder)
            lbl = 0 if folder.startswith("Fresh") else 1
            for fname in _list_images(fp):
                key = (split, folder, fname)
                if only_ok and key in escamas:
                    n_escamas += 1
                    continue
                if only_ok and estados.get(key) != "ok":
                    if key in rescatadas:
                        n_rescatadas += 1
                    else:
                        n_excluidas += 1
                        continue
                paths.append(os.path.join(fp, fname))
                labels.append(lbl)
    if only_ok:
        print(f"[collect_rohu_eyes] excluidas {n_excluidas} 'dudosa' + incluidas {n_rescatadas} rescatadas "
              f"manualmente + excluidas {n_escamas} falsos positivos confirmados (escama/aleta/borrosa)")
    return paths, labels


DOMAINS = {
    "mendeley": {"nombre": "Mendeley Data (fuente)", "loader": collect_mendeley, "rol": "fuente"},
    "onedrive_trucha": {"nombre": "OneDrive - Trucha ojos, preprocesado (objetivo 1)",
                        "loader": collect_trucha, "rol": "objetivo"},
    "kaggle_rohu": {"nombre": "Kaggle - Rohu ojos, preprocesado (objetivo 2)",
                    "loader": collect_rohu_eyes, "rol": "objetivo"},
}


if __name__ == "__main__":
    from PIL import Image
    for key, d in DOMAINS.items():
        paths, labels = d["loader"]()
        n0 = labels.count(0)
        n1 = labels.count(1)
        size0 = Image.open(paths[0]).size if paths else None
        print(f"{key:16s} ({d['nombre']}): {len(paths)} imagenes  fresco={n0}  no_fresco={n1}  "
              f"resolucion ejemplo={size0}")
