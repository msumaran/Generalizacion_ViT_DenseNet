"""Detecta imagenes casi-duplicadas (fotos en rafaga del mismo ojo) en
Trucha/Rohu y verifica si el 5-fold CV usado en finetune_target_x2.py las
reparte entre train y validacion de un mismo fold -- si un par casi-
identico cae uno en train y otro en val, el modelo "memoriza" en vez de
generalizar y el F1 de validacion deja de ser confiable.

Se origino al ver F1=100.00 en Rohu en fine-tuning en casi todos los folds
de los 4 modelos (incluido DenseNet-121, que en zero-shot era el peor) --
una senal demasiado limpia y uniforme entre arquitecturas para ser
generalizacion real.

Metodo: average hash (8x8 escala de grises, umbral = media) + distancia de
Hamming. Barato, suficiente para detectar cuasi-duplicados en fotos ya
recortadas/normalizadas (330x330, mismo CLAHE) -- no pretende ser un
detector de duplicados de proposito general.

Uso: python detectar_duplicados.py [--umbral 2] [--dominio onedrive_trucha|kaggle_rohu|todos]
"""
import argparse

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold

from domains import DOMAINS

N_FOLDS = 5
SEED = 42


def ahash(path, size=8):
    im = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    arr = np.asarray(im, dtype=np.float64)
    return (arr > arr.mean()).flatten()


def grupos_casi_duplicados(paths, umbral=2):
    """Clustering jerarquico *complete-linkage* sobre distancia de Hamming
    del average-hash: devuelve un array de ids de grupo (misma longitud que
    paths). Pensado para pasarse como `groups` a GroupKFold/
    StratifiedGroupKFold y asi evitar que un fold entrene con una foto y
    valide con su casi-duplicado (ver finetune_target_x2.py).

    Se probo primero con union-find (single-linkage: A~B~C se encadenan
    aunque A y C no se parezcan) y en Rohu formaba UN grupo de 481/1559
    imagenes (31% del dataset) -- el hash de 8x8 es demasiado grueso para
    este dataset (todos los ojos de Rohu lucen estructuralmente parecidos:
    pupila oscura + anillo naranja + fondo de escama) y encadenaba fotos
    genuinamente distintas. Complete-linkage exige que TODOS los miembros
    de un grupo esten a <= umbral entre si (no solo el vecino inmediato),
    lo que evita esa cadena mientras sigue agrupando duplicados reales
    (verificado: mismo grupo para una rafaga confirmada visualmente,
    IMG_0821/0822/0823.JPG en Rohu) y deja folds razonablemente
    balanceados (17-24% cada uno, vs. un fold de 5.6% con union-find)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    n = len(paths)
    hashes = np.array([ahash(p) for p in paths], dtype=np.float64)
    D = pdist(hashes, metric="hamming") * hashes.shape[1]  # de vuelta a conteo de bits
    Z = linkage(D, method="complete")
    grupos = fcluster(Z, t=umbral, criterion="distance") - 1  # 0-indexado

    tam_grupo = np.bincount(grupos)
    print(f"[grupos_casi_duplicados] {n} imagenes -> {len(tam_grupo)} grupos "
          f"({int((tam_grupo > 1).sum())} grupos con 2+ imagenes casi-identicas, "
          f"grupo mas grande={int(tam_grupo.max())})")
    return grupos


def analizar(domain_key, umbral):
    paths, labels = DOMAINS[domain_key]["loader"]()
    labels = np.array(labels)
    n = len(paths)
    hashes = np.array([ahash(p) for p in paths])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_of = np.zeros(n, dtype=int)
    for fi, (_, val_idx) in enumerate(skf.split(np.zeros(n), labels)):
        fold_of[val_idx] = fi

    total_pares = 0
    fuga_pares = 0
    ejemplos = []
    for i in range(n):
        dists = (hashes[i] != hashes[i + 1:]).sum(axis=1)
        cerca = np.where(dists <= umbral)[0]
        for c in cerca:
            j = i + 1 + c
            total_pares += 1
            if fold_of[i] != fold_of[j]:
                fuga_pares += 1
                if len(ejemplos) < 10:
                    ejemplos.append((paths[i], paths[j], int(dists[c]), int(fold_of[i]), int(fold_of[j])))

    print(f"\n{domain_key}: n={n}  umbral_hamming<={umbral}")
    print(f"  pares casi-duplicados (total): {total_pares}")
    print(f"  de esos, en folds DISTINTOS (fuga train/val real en el 5-fold CV): {fuga_pares}")
    if fuga_pares:
        print("  ejemplos:")
        for p in ejemplos:
            print(f"    {p[0]}  <->  {p[1]}   dist={p[2]}  fold{p[3]} vs fold{p[4]}")
    return total_pares, fuga_pares


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral", type=int, default=2,
                     help="distancia de Hamming maxima (de 64 bits) para considerar casi-duplicado")
    ap.add_argument("--dominio", choices=["onedrive_trucha", "kaggle_rohu", "todos"], default="todos")
    args = ap.parse_args()

    dominios = ["onedrive_trucha", "kaggle_rohu"] if args.dominio == "todos" else [args.dominio]
    for d in dominios:
        analizar(d, args.umbral)
