"""Genera la figura de arquitectura del sistema (pipeline cross-dataset congelado, Trucha y Rohu)
usada en Documentacion_Tecnica_ViT_DenseNet.docx, seccion 1.3. Misma estructura que la figura de
Mendeley_ViT_DenseNet/1_codigo/generar_arquitectura_sistema.py.

Estructura en 4 etapas (ingesta -> entrenamiento -> evaluacion -> reportes),
con las fuentes de datos, el almacenamiento local y los entregables abajo.

Uso (desde la raiz del repo; requiere matplotlib):
    python 1_codigo/generar_arquitectura_sistema.py
Salida: 4_graficas/arquitectura_sistema.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle

OUT = "4_graficas/arquitectura_sistema.png"

NARANJA, NARANJA_BORDE = "#F4B183", "#E07B39"
AZUL, AZUL_BORDE = "#DAE6F4", "#6F8FBF"
GRIS_MARCO, GRIS_BARRA, GRIS_CAJA = "#7F7F7F", "#A6A6A6", "#9E9E9E"
TITULO = "#E8642C"

fig, ax = plt.subplots(figsize=(16, 9.4))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9.4)
ax.axis("off")


def caja(x, y, w, h, texto, fc, ec, size=9, bold=False, color="black"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal", color=color, linespacing=1.25)


def flecha(puntos, color="black", doble=False, lw=1.1):
    """Linea con codos; la punta va en el ultimo tramo."""
    for (x0, y0), (x1, y1) in zip(puntos[:-2], puntos[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw)
    estilo = "<|-|>" if doble else "-|>"
    ax.add_patch(FancyArrowPatch(puntos[-2], puntos[-1], arrowstyle=estilo, mutation_scale=12,
                                 color=color, lw=lw))


def circulo_preparar(cx, cy):
    ax.add_patch(Circle((cx, cy), 0.28, facecolor="white", edgecolor=AZUL_BORDE, lw=1.3))
    ax.plot([cx - 0.28, cx + 0.28], [cy, cy], color=AZUL_BORDE, lw=1.1)
    ax.plot([cx, cx], [cy - 0.28, cy + 0.28], color=AZUL_BORDE, lw=1.1)


# --- barra lateral -----------------------------------------------------------
ax.add_patch(Rectangle((0.2, 3.25), 1.35, 5.05, facecolor=GRIS_BARRA, edgecolor="none"))
ax.text(0.875, 6.15, "Entorno\nde Data\nScience", ha="center", va="center", fontsize=11.5,
        fontweight="bold", color="white", linespacing=1.3)
ax.text(0.875, 4.7, "Python 3.9\nPyTorch\ntorchvision\ntimm\nscikit-learn\npandas\nmatplotlib",
        ha="center", va="center", fontsize=8.5, color="white", linespacing=1.35)

# --- 4 columnas --------------------------------------------------------------
COLS = [  # (x, titulo)
    (1.85, "Ingesta y\npreparación de datos"),
    (5.35, "Entrenamiento\nde modelos"),
    (8.85, "Evaluación y\nselección de modelos"),
    (12.35, "Reportes y\ndocumentación"),
]
W, Y0, H = 3.2, 3.25, 5.05
for x, titulo in COLS:
    ax.add_patch(Rectangle((x, Y0), W, H, facecolor="white", edgecolor=GRIS_MARCO, lw=2.2))
    ax.text(x + W / 2, 8.85, titulo, ha="center", va="center", fontsize=11.5, fontweight="bold",
            color=TITULO, linespacing=1.15)

# Columna 1: ingesta
x = COLS[0][0]
caja(x + 0.35, 6.55, 2.5, 1.35, "Definición del\ntratamiento del dato\n(etiquetas fresco /\nno_fresco, revisión\nmanual de recortes)",
     AZUL, AZUL_BORDE, size=8.8)
caja(x + 0.35, 3.75, 2.5, 1.6, "Módulo de partición\nTrucha: StratifiedKFold\nRohu: StratifiedGroupKFold\n10 folds (seed 42)",
     AZUL, AZUL_BORDE, size=8.8)
flecha([(x + 1.6, 6.5), (x + 1.6, 5.4)], color=GRIS_CAJA, doble=True)

# Columnas 2-4: cabecera + linea punteada + "preparar" + pipeline
CABECERAS = [
    ("Pipeline de\nentrenamiento", NARANJA, NARANJA_BORDE, "black"),
    ("Pipeline de\nevaluación", GRIS_CAJA, GRIS_CAJA, "white"),
    ("Pipeline de\nreportes", NARANJA, NARANJA_BORDE, "black"),
]
SCRIPTS = ["linear_probe_con_test.py", "linear_probe_con_test.py",
           "generar_documentacion_tecnica.py"]
for (x, _), (txt, fc, ec, col), script in zip(COLS[1:], CABECERAS, SCRIPTS):
    caja(x + 0.55, 7.35, 2.1, 0.7, txt, fc, ec, size=9.5, bold=True, color=col)
    ax.text(x + W / 2, 7.12, script, ha="center", va="center", fontsize=7.6, style="italic", color="#444444")
    ax.plot([x + 0.1, x + W - 0.1], [6.9, 6.9], ls="--", color=GRIS_MARCO, lw=1)

# Columna 2: entrenamiento
x = COLS[1][0]
ax.text(x + 1.6, 6.6, "Preparar datos", ha="center", fontsize=9.5)
circulo_preparar(x + 1.6, 6.0)
ax.text(x + 2.0, 6.0, "resize +\naugmentation", ha="left", va="center", fontsize=7.4, color="#444444")
flecha([(x + 1.6, 5.7), (x + 1.6, 5.3)], color=GRIS_CAJA)
caja(x + 0.55, 4.25, 2.1, 1.05, "Linear probe (congelado)\n2 modelos × 2 dominios\n× 10 folds · AdamW", NARANJA,
     NARANJA_BORDE, size=8.6, bold=True)
ax.text(x + 1.6, 3.7, "Entrenar la cabeza y\nelegir el mejor epoch\n(val_F1)", ha="center", va="center",
        fontsize=9, fontweight="bold", linespacing=1.2)

# Columna 3: evaluacion
x = COLS[2][0]
ax.text(x + 1.6, 6.6, "MODEL", ha="center", fontsize=8.5, fontweight="bold")
for i, h in enumerate([0.22, 0.38, 0.3, 0.5]):  # icono: barras + lupa
    ax.add_patch(Rectangle((x + 1.25 + i * 0.13, 5.85), 0.08, h, color="#333333"))
ax.add_patch(Circle((x + 1.88, 6.18), 0.15, facecolor="none", edgecolor="#333333", lw=1.6))
ax.plot([x + 1.98, x + 2.12], [6.07, 5.93], color="#333333", lw=2)
ax.text(x + 1.6, 5.6, "Evaluation", ha="center", fontsize=8.5, fontweight="bold")
flecha([(x + 1.6, 5.45), (x + 1.6, 5.3)], color=GRIS_CAJA)
caja(x + 0.35, 4.25, 2.5, 1.05, "Inferencia sobre el fold\nmatriz de confusión · ROC\n· AUC · pooled OOF",
     NARANJA, NARANJA_BORDE, size=8.6, bold=True)
ax.text(x + 1.6, 3.7, "Indicadores de robustez:\nTRD · Gap · RBE", ha="center", va="center",
        fontsize=9, fontweight="bold", linespacing=1.2)

# Columna 4: reportes
x = COLS[3][0]
ax.text(x + 1.6, 6.6, "Preparar resultados", ha="center", fontsize=9.5)
circulo_preparar(x + 1.6, 6.0)
flecha([(x + 1.6, 5.7), (x + 1.6, 5.4)], color=GRIS_CAJA)
caja(x + 0.55, 4.6, 2.1, 0.8, "Gráficas y tablas\n(matrices, ROC, F1)", NARANJA, NARANJA_BORDE, size=8.6, bold=True)
flecha([(x + 1.6, 4.55), (x + 1.6, 4.25)], color=GRIS_CAJA)
caja(x + 0.55, 3.5, 2.1, 0.75, "Carpetas de\nresultados (PNG, CSV)", AZUL, AZUL_BORDE, size=8.6)

# --- fila inferior -----------------------------------------------------------
# Fuentes de datos: pila de imagenes
for i, (dx, dy) in enumerate([(0.0, 0.0), (0.12, 0.12), (0.24, 0.24)]):
    ax.add_patch(Rectangle((2.2 + dx, 1.05 + dy), 0.9, 0.7, facecolor="white", edgecolor=GRIS_MARCO, lw=1))
ax.add_patch(Rectangle((2.5, 1.38), 0.5, 0.42, facecolor=AZUL, edgecolor=AZUL_BORDE, lw=0.8))
ax.text(3.65, 1.55, "Trucha: 1,150\nRohu: 1,261\nimágenes\n+ checkpoints de\nMendeley (5 folds)",
        ha="left", va="center", fontsize=8.3, linespacing=1.2)
flecha([(3.45, 2.45), (3.45, 3.2)], color=NARANJA_BORDE, doble=True, lw=3)
ax.text(3.45, 0.35, "Fuentes de datos", ha="center", fontsize=11, fontweight="bold", color=TITULO)

# Almacenamiento local
caja(6.3, 0.95, 4.5, 1.3, "Almacenamiento local\ncheckpoints de Mendeley (.pt) · métricas CSV\n· logs de entrenamiento",
     AZUL, AZUL_BORDE, size=9)
ax.text(8.55, 0.35, "Datos y modelos organizados en carpetas", ha="center", fontsize=11,
        fontweight="bold", color=TITULO)

# Entregables
caja(12.35, 0.95, 3.2, 1.3, "Informe de tesis\nDocumentación técnica ·\nIndicadores de robustez", AZUL, AZUL_BORDE, size=9)
ax.text(13.95, 0.35, "Resultados documentados", ha="center", fontsize=11, fontweight="bold", color=TITULO)

# --- flujos (ningun tramo comparte linea con otro) --------------------------
x1, x2, x3, x4 = (c[0] for c in COLS)
flecha([(x1 + 2.85, 4.4), (x1 + 3.3, 4.4), (x1 + 3.3, 1.6), (6.3, 1.6)])               # particion -> almacen
flecha([(6.5, 2.25), (6.5, 2.8), (x2 + 0.25, 2.8), (x2 + 0.25, 6.0), (x2 + 1.32, 6.0)])  # almacen -> entrenamiento
flecha([(x2 + 1.6, 3.4), (x2 + 1.6, 2.25)])                                             # checkpoints -> almacen
flecha([(8.6, 2.25), (8.6, 2.8), (x3 + 0.2, 2.8), (x3 + 0.2, 4.75), (x3 + 0.35, 4.75)])  # almacen -> evaluacion
flecha([(x3 + 1.6, 3.4), (x3 + 1.6, 2.25)])                                             # metricas -> almacen
flecha([(10.8, 1.95), (11.95, 1.95), (11.95, 6.0), (x4 + 1.32, 6.0)])                    # almacen -> reportes
flecha([(x4 + 1.6, 3.5), (x4 + 1.6, 2.25)], color=GRIS_CAJA)                              # resultados -> informe
flecha([(10.8, 1.3), (12.35, 1.3)], color=GRIS_CAJA)                                      # almacen -> informe

fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Guardado en {OUT}")
