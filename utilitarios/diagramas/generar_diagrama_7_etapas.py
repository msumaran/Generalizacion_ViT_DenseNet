"""Diagramas del experimento de generalización (Trucha y Rohu), con el estilo común de
herramientas_docx/diagramas_svg.py.

Genera en 6_documentacion/diagramas/ (SVG editable + PNG en alta resolución):
  - arquitectura_generalizacion_7_etapas: selección de imágenes y seis etapas (el resultado se explica aparte);
  - vista_simple_generalizacion: el experimento en 6 pasos, para público general.

Cifras: Trucha 1,196 → 1,150 (46 excluidas); Rohu 1,652 → 1,521 depuradas, 1,261 evaluadas (ver la
Nota sobre Rohu); TRD de ViT-Base/16 y DenseNet-121: 89.09 / 86.43 (Trucha) y 92.42 / 88.07 (Rohu); 10 folds.
Uso (desde la raíz del repositorio): python utilitarios/diagramas/generar_diagrama_7_etapas.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent / "herramientas_docx"))
import diagramas_svg as d  # noqa: E402

OUT = ROOT / "6_documentacion" / "diagramas"
NARANJA, OCRE, AZUL, MAGENTA, VERDE = "#dd861a", "#c8930b", "#5154a1", "#a42b69", "#148f87"
ANCHO, ALTO = d.X0 + 7 * d.COL, 1650


# Tipo de cada caja (ícono del diagrama); la documentación lo usa para nombrar cada componente
# Orden de lectura: el número de cada caja coincide con la explicación del documento
ORDEN_CAJAS = ["Selección de trucha", "Selección de Rohu", "Revisión de trucha", "Fotos listas", "Modelo de Mendeley", "Dividir en 10 folds", "Entrenar la cabeza", "Probar en el fold", "Guardar y decidir", "Evaluar el fold", "Reunir 10 folds", "Gráficas", "Comparar", "Documentación"]
TIPO_CAJA = {
    "Selección de trucha": "dataset",
    "Selección de Rohu": "dataset",
    "Revisión de trucha": "función",
    "Fotos listas": "dataset",
    "Modelo de Mendeley": "archivo",
    "Dividir en 10 folds": "función",
    "Entrenar la cabeza": "componente",
    "Probar en el fold": "función",
    "Guardar y decidir": "función",
    "Evaluar el fold": "función",
    "Reunir 10 folds": "función",
    "Gráficas": "archivo",
    "Comparar": "función",
    "Documentación": "archivo",
}


def arquitectura():
    L = d.Lienzo(ANCHO, ALTO)
    caja = lambda c, f, color, titulo, lineas, **kw: L.caja(  # noqa: E731
        c, f, color, f"{ORDEN_CAJAS.index(titulo) + 1}. {titulo}", lineas, tipo=TIPO_CAJA.get(titulo), **kw)
    for c in range(7):
        L.panel(c, 1030)
    cabeceras = [(d.GRIS, "ANTES DEL EXPERIMENTO", "Seleccionar imágenes"), (NARANJA, "ETAPA 1", "Preparar datos"),
                 (OCRE, "ETAPA 2", "Preparar experimento"), (AZUL, "ETAPA 3", "Entrenar la cabeza"),
                 (MAGENTA, "ETAPA 4", "Evaluar"), (MAGENTA, "ETAPA 5", "Reunir resultados"),
                 (VERDE, "ETAPA 6", "Comparar y documentar")]
    for c, (color, l1, l2) in enumerate(cabeceras):
        L.cabecera(c, color, l1, l2)

    caja(0, 0, d.GRIS, "Selección de trucha", ["OneDrive (Perú)", "Fresco: días 1 y 3", "No fresco: días 5, 7 y 9",
                                                 "1,196 fotos"])
    caja(0, 1, d.GRIS, "Selección de Rohu", ["Kaggle (India)", "1,652 fotos de ojos", "Se excluyen 131", "Quedan 1,521"])
    caja(1, 0, NARANJA, "Revisión de trucha", ["128 recortes dudosos", "revisados a mano", "Se excluyen 46"])
    caja(1, 1, NARANJA, "Fotos listas", ["Trucha: 1,150", "Rohu: 1,261", "Fresco / no fresco"])
    caja(2, 0, OCRE, "Modelo de Mendeley", ["5 checkpoints por modelo", "ViT-Base/16 o", "DenseNet-121"])
    caja(2, 1, OCRE, "Dividir en 10 folds", ["Arma 10 folds (una vez)", "En cada vuelta:", "1 fold → evaluar",
                                             "los otros 9 → entrenar", "Rohu: ráfagas juntas"])
    caja(3, 0, AZUL, "Entrenar la cabeza", ["Modelo congelado", "Solo cambia la última capa",
                                              "Una pasada por los otros 9", "Continúa desde la anterior",
                                              "Variaciones al azar"])
    caja(3, 1, AZUL, "Probar en el fold", ["Clasifica el fold", "que quedó fuera", "(el mismo en cada época)",
                                           "y mide el F1"])
    caja(3, 2, AZUL, "Guardar y decidir", ["Si el F1 mejora, guarda", "la cabeza de esa época", "Hasta completar", "30 épocas"])
    caja(4, 0, MAGENTA, "Evaluar el fold", ["Cabeza de la mejor época", "clasifica el fold", "Matriz · ROC · AUC"])
    caja(5, 0, MAGENTA, "Reunir 10 folds", ["Cada foto evaluada", "una sola vez", "F1 y AUC por modelo"])
    caja(5, 1, MAGENTA, "Gráficas", ["Matrices de confusión", "Curvas ROC"])
    caja(6, 0, VERDE, "Comparar", ["Cuánto conserva: TRD", "Cuánto pierde: Gap", "ViT frente a CNN: RBE"])
    caja(6, 1, VERDE, "Documentación", ["Tablas, gráficas", "y registros", "en este documento"])

    cx = lambda c: d.x_caja(c) + d.ANCHO / 2  # noqa: E731
    yc = lambda f: d.FILAS_Y[f] + d.ALTO / 2  # noqa: E731
    der = lambda c: d.x_caja(c) + d.ANCHO  # noqa: E731
    izq = d.x_caja
    arriba = lambda f: d.FILAS_Y[f]  # noqa: E731
    abajo = lambda f: d.FILAS_Y[f] + d.ALTO  # noqa: E731
    G = d.GRIS

    # Antes del experimento → etapa 1
    L.flecha([(der(0), yc(0)), (izq(1), yc(0))], G); L.etiqueta(d.hueco(0), yc(0) - 30, "1,196", G)
    L.flecha([(der(0), yc(1)), (izq(1), yc(1))], G); L.etiqueta(d.hueco(0), yc(1) - 30, "1,261", G)
    L.flecha([(cx(1), abajo(0)), (cx(1), arriba(1))], NARANJA)
    L.etiqueta(cx(1) + 14, d.franja(0), "quedan 1,150", NARANJA, "start")
    # Etapa 1 → 2
    L.flecha([(der(1), yc(1)), (izq(2), yc(1))], NARANJA); L.etiqueta(d.hueco(1), yc(1) - 30, "fotos", NARANJA)
    L.flecha([(cx(2), d.CARRIL_SUPERIOR), (cx(2), arriba(0))], OCRE)
    L.etiqueta(cx(2), d.CARRIL_SUPERIOR, "viene del experimento Mendeley", OCRE)
    # Etapa 2 → 3: el modelo y los 4 folds van a entrenar; el fold de evaluación, a elegir la época
    L.flecha([(der(2), yc(0)), (izq(3), yc(0))], OCRE); L.etiqueta(d.hueco(2), yc(0) - 30, "modelo", OCRE)
    L.flecha([(der(2), yc(1) - 50), (d.hueco(2), yc(1) - 50), (d.hueco(2), yc(0) + 60), (izq(3), yc(0) + 60)], OCRE)
    L.etiqueta(d.hueco(2) - 12, d.franja(0), "los\notros 9", OCRE, "end")
    L.flecha([(der(2), yc(1) + 50), (izq(3), yc(1) + 50)], OCRE)
    L.etiqueta(d.hueco(2) - 14, yc(1) + 20, "1 fold", OCRE)
    # Ciclo por época: entrenar → probar en el fold de evaluación → elegir; si no se detiene, otra época
    L.flecha([(cx(3), abajo(0)), (cx(3), arriba(1))], AZUL)
    L.etiqueta(cx(3) + 14, d.franja(0), "cabeza\nentrenada", AZUL, "start")
    L.flecha([(cx(3), abajo(1)), (cx(3), arriba(2))], AZUL)
    L.etiqueta(cx(3) - 14, d.franja(1), "F1 de\nla época", AZUL, "end")
    # Etapa 3 → 4 → 5 → 6 → resultado
    L.flecha([(der(3), yc(2)), (d.hueco(3), yc(2)), (d.hueco(3), yc(0) + 50), (izq(4), yc(0) + 50)], AZUL)
    L.etiqueta(d.hueco(3) + 12, d.franja(0) + 40, "al terminar:\nmejor época", AZUL, "start")
    L.flecha([(der(4), yc(0)), (izq(5), yc(0))], MAGENTA)
    L.flecha([(cx(5), abajo(0)), (cx(5), arriba(1))], MAGENTA); L.etiqueta(cx(5) + 14, d.franja(0), "datos", MAGENTA, "start")
    L.flecha([(der(5), yc(0)), (izq(6), yc(0))], MAGENTA)
    # La comparación necesita también el F1 que el modelo logró en Mendeley
    L.flecha([(cx(6), d.CARRIL_SUPERIOR), (cx(6), arriba(0))], VERDE)
    L.etiqueta(cx(6), d.CARRIL_SUPERIOR, "F1 en Mendeley: 97.04 y 95.66", VERDE)
    L.flecha([(der(5), yc(1)), (izq(6), yc(1))], MAGENTA); L.etiqueta(d.hueco(5), yc(1) - 30, "figuras", MAGENTA)
    L.flecha([(cx(6), abajo(0)), (cx(6), arriba(1))], VERDE); L.etiqueta(cx(6) + 14, d.franja(0), "indicadores", VERDE, "start")

    # Registros (punteado)
    yb = 1330
    L.banda(3, 6, yb, 110, "15. Registros y resultados guardados",
            "Entrenamiento por época, resultados por fold y gráficas de la corrida de 10 folds")
    L.flecha([(cx(3), abajo(2)), (cx(3), yb)], d.LINEA, True)
    L.etiqueta(cx(3) + 14, abajo(2) + 40, "registro por época", d.LINEA, "start")
    L.flecha([(cx(4), abajo(0)), (cx(4), yb)], d.LINEA, True)
    L.etiqueta(cx(4) + 14, abajo(1) + 120, "resultados por fold", d.LINEA, "start")
    L.flecha([(cx(6), yb), (cx(6), abajo(1))], d.LINEA, True)
    L.etiqueta(cx(6) + 14, abajo(1) + 120, "respaldo", d.LINEA, "start")

    L.leyenda(1470, ["Flecha continua: paso de datos de una caja a otra. Flecha punteada: registro que se guarda o se "
                     "consulta. Cada color es una etapa; la columna gris ocurre antes del experimento.",
                     "Cabeza: la última capa del modelo, la que decide fresco o no fresco; es lo único que se vuelve a "
                     "entrenar. Fold: cada una de las 10 partes en que se dividen las fotos."])
    svg = L.svg("GENERALIZACIÓN A TRUCHA Y ROHU · ARQUITECTURA DEL SISTEMA",
                "Los modelos entrenados con Mendeley se prueban con fotos de otros peces: selección de imágenes, "
                "seis etapas")
    d.guardar(svg, OUT, "arquitectura_generalizacion_7_etapas")


def vista_simple():
    d.guardar(d.vista_simple(
        "Generalización a otros peces en 6 pasos",
        "Pregunta: ¿cuánto de lo aprendido con Mendeley sirve con fotos de otros peces?",
        [(OCRE, "Partir de Mendeley", "2 modelos", "ViT-Base/16 y\nDenseNet-121,\nya entrenados"),
         (NARANJA, "Fotos nuevas", "Trucha 1,150\nRohu 1,261", "peces de Perú\ne India"),
         (AZUL, "Ajuste mínimo", "1 capa", "solo se reentrena\nla última capa"),
         (MAGENTA, "Examen", "10 folds", "cada foto se evalúa\nuna sola vez"),
         (VERDE, "Comparar", "vs Mendeley", "¿cuánto conserva\ncada modelo?"),
         (d.DORADO, "Resultado", "ViT-Base/16", "conserva más:\n89.0 % vs 86.5 % (Trucha)\n93.0 % vs 88.0 % (Rohu)")]),
        OUT, "vista_simple_generalizacion")


if __name__ == "__main__":
    arquitectura()
    vista_simple()
