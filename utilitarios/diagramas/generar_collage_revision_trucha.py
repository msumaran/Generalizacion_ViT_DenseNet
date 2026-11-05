"""Collage de la revisión de Trucha: recortes «ok» conservados, «dudosa» rescatados a mano y excluidos.

Las imágenes no están en el repositorio: correr en el equipo que tiene ojos_procesados/ y
1_codigo/verificacion_preprocesamiento/trucha_rescatadas.csv, desde la carpeta que contiene ojos_procesados/
(las rutas de domains.py son relativas a esa carpeta):

    cd <carpeta con ojos_procesados>
    python <repo>/utilitarios/diagramas/generar_collage_revision_trucha.py --motivos motivos.csv

--motivos (opcional): CSV con columnas clase,archivo,motivo para las excluidas. Si se pasa, el collage
muestra solo excluidas con motivo; si no, se rotulan «excluida».
Salida: <repo>/4_graficas/revision_trucha/collage_revision_trucha.png (la documentación la inserta si existe).
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "1_codigo"))
from domains import TRUCHA_DIR, _load_trucha_estados, _load_trucha_rescatadas  # noqa: E402

COLUMNAS = 6
SALIDA = REPO / "4_graficas" / "revision_trucha" / "collage_revision_trucha.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motivos")
    args = ap.parse_args()
    estados, rescatadas = _load_trucha_estados(), _load_trucha_rescatadas()
    assert estados and rescatadas, "faltan deteccion_scores.csv o trucha_rescatadas.csv (ver docstring)"
    ok = sorted(k for k, e in estados.items() if e == "ok")
    excluidas = sorted(k for k, e in estados.items() if e != "ok" and k not in rescatadas)
    assert len(excluidas) == 46, len(excluidas)  # cifra del documento
    motivos = {}
    if args.motivos:
        with open(args.motivos) as f:
            motivos = {(r["clase"], r["archivo"]): r["motivo"] for r in csv.DictReader(f)}
        excluidas = [k for k in excluidas if k in motivos]
        assert excluidas, "ninguna excluida tiene motivo"
    rng = random.Random(42)
    filas = [("Conservada («ok»)", rng.sample(ok, COLUMNAS), {}),
             ("Rescatada a mano", rng.sample(sorted(rescatadas), COLUMNAS), {}),
             ("Excluida", rng.sample(excluidas, min(COLUMNAS, len(excluidas))), motivos)]

    fig, ejes = plt.subplots(3, COLUMNAS, figsize=(2.2 * COLUMNAS, 7.6))
    for (titulo, claves, rotulos), fila in zip(filas, ejes):
        for i, ax in enumerate(fila):
            ax.axis("off")
            if i >= len(claves):
                continue
            clase, archivo = claves[i]
            ax.imshow(Image.open(Path(TRUCHA_DIR) / clase / archivo).convert("RGB"))
            pie = rotulos.get(claves[i], clase.replace("_", " "))
            ax.set_title(pie, fontsize=8, wrap=True)
        fila[0].text(-0.08, 0.5, titulo, transform=fila[0].transAxes, rotation=90,
                     ha="right", va="center", fontsize=10, weight="bold")
    fig.tight_layout()
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA, dpi=200, bbox_inches="tight")
    print(f"Guardado: {SALIDA}")


if __name__ == "__main__":
    main()
