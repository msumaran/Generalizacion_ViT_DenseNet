"""Agrega al documento técnico de generalización los ajustes acordados en la reunión del 30/10/2026:
explicación propia por etapa, explicación de la arquitectura caja por caja, depuración de trucha
en dos pasos (homologación 3 → 2 estados y exclusión por calidad), nota final sobre la composición de Rohu,
justificación del protocolo de folds, cómo se obtiene cada columna de la tabla general y por qué
algunos folds terminan antes de 30 epochs.

Los textos nuevos van sombreados (amarillo) para ubicarlos rápido; lo que falta confirmar va
en rojo como [PENDIENTE: ...]. Uso (desde la raíz del repositorio): python utilitarios/documentacion/agregar_explicaciones_reunion.py
"""
import copy
import re
import shutil
import struct
import unicodedata
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml.ns import qn
from docx.shared import Cm, Emu, Pt, RGBColor
from docx.table import _Cell
from docx.text.paragraph import Paragraph
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagramas"))
from generar_diagrama_7_etapas import TIPO_CAJA  # noqa: E402  (tipo de cada caja, igual que en el diagrama)

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "6_documentacion" / "Documentacion_Tecnica_ViT_DenseNet.docx"
FIGURA = ROOT / "6_documentacion" / "diagramas" / "arquitectura_generalizacion_7_etapas.png"
MARCA = "Explicación del autor"
SOMBRA = "FFF2CC"


def _shade(p):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), SOMBRA)
    # w:shd va antes de estos hijos de w:pPr (orden del esquema de Word)
    p._p.get_or_add_pPr().insert_element_before(
        shd, "w:tabs", "w:suppressAutoHyphens", "w:kinsoku", "w:wordWrap", "w:overflowPunct",
        "w:topLinePunct", "w:autoSpaceDE", "w:autoSpaceDN", "w:bidi", "w:adjustRightInd",
        "w:snapToGrid", "w:spacing", "w:ind", "w:contextualSpacing", "w:mirrorIndents",
        "w:suppressOverlap", "w:jc", "w:textDirection", "w:textAlignment",
        "w:textboxTightWrap", "w:outlineLvl", "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr",
        "w:pPrChange")


class Cursor:
    """Inserta párrafos y tablas en orden, a continuación de un elemento de referencia."""

    def __init__(self, doc, ref):
        self.doc, self.el, self.parent = doc, ref._p, ref._parent

    def p(self, *runs, style=None, shade=False):
        el = OxmlElement("w:p")
        self.el.addnext(el)
        self.el = el
        par = Paragraph(el, self.parent)
        if style:
            par.style = self.doc.styles[style]
        for item in runs:
            text, kind = (item, None) if isinstance(item, str) else item
            r = par.add_run(text)
            r.bold = kind == "b"
            if kind == "r":
                r.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
                r.bold = True
        if shade and not style:
            _shade(par)
        return par

    def f(self, *formulas):
        """Líneas de fórmula con el formato del documento (Consolas, con sangría)."""
        for texto in formulas:
            par = self.p(texto, shade=False)
            par.runs[0].font.name = "Consolas"
            par.paragraph_format.left_indent = Emu(215900)

    def table(self, rows, bold_last=False):
        """Tabla con el mismo formato que las del documento: encabezado azul, texto blanco."""
        t = self.doc.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = self.doc.styles["Table Grid"]
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                cell = t.cell(i, j)
                cell.text = str(val)
                par = cell.paragraphs[0]
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = par.runs[0]
                run.font.size = Pt(9.5)
                run.bold = i == 0 or (bold_last and i == len(rows) - 1)
                if i == 0:
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    shd = OxmlElement("w:shd")
                    shd.set(qn("w:val"), "clear")
                    shd.set(qn("w:color"), "auto")
                    shd.set(qn("w:fill"), "2A4D7A")
                    cell._tc.get_or_add_tcPr().append(shd)
        self.el.addnext(t._tbl)
        self.el = t._tbl
        return t


def find(doc, prefix):
    return next(p for p in doc.paragraphs if p.text.strip().startswith(prefix))


def _tamano_png(datos):
    """(ancho, alto) en píxeles, leídos de la cabecera IHDR del PNG."""
    return struct.unpack(">II", datos[16:24])


def reemplazar_imagen(doc, blip, ruta):
    """Cambia la imagen de un dibujo y ajusta su alto a la nueva proporción (mismo ancho)."""
    datos = ruta.read_bytes()
    doc.part.related_parts[blip.get(qn("r:embed"))]._blob = datos
    ancho, alto = _tamano_png(datos)
    inline = next(a for a in blip.iterancestors() if a.tag in (qn("wp:inline"), qn("wp:anchor")))
    extent = inline.find(qn("wp:extent"))
    nuevo_cy = str(round(int(extent.get("cx")) * alto / ancho))
    extent.set("cy", nuevo_cy)
    for ext in inline.iter(qn("a:ext")):
        if ext.get("cy") is not None:
            ext.set("cy", nuevo_cy)


def quitar(p):
    p._p.getparent().remove(p._p)


def reemplazar_texto(doc, viejo, nuevo):
    """Reemplaza viejo por nuevo en los runs del cuerpo y de las tablas; devuelve cuántos cambió.
    Solo asigna r.text si el run contiene el texto (asignarlo borra imágenes del run)."""
    parrafos = list(doc.paragraphs) + [p for t in doc.tables for c in t._cells for p in c.paragraphs]
    n = 0
    for p in parrafos:
        for r in p.runs:
            if viejo in r.text:
                r.text = r.text.replace(viejo, nuevo)
                n += 1
    return n


def reemplazar_texto_celda(celda, viejo, nuevo):
    n = 0
    for p in celda.paragraphs:
        for r in p.runs:
            if viejo in r.text:
                r.text = r.text.replace(viejo, nuevo)
                n += 1
    return n


def quitar_aclaracion_f1(doc):
    """La aclaración «la columna F1 es siempre el F1 macro» ya no aplica: las tablas por fold dicen
    «F1 macro» y en las tablas por clase «F1» es el de esa clase."""
    titulo = find(doc, "Aclaración importante")
    quitar(Paragraph(titulo._p.getnext(), titulo._parent))
    quitar(titulo)


def quitar_del_glosario(doc, termino):
    tabla = next(t for t in doc.tables if t.cell(0, 0).text.strip() == "Término")
    fila = next(f for f in tabla.rows if f.cells[0].text.strip() == termino)
    fila._tr.getparent().remove(fila._tr)


def enlace_ampliado(doc, despues_de, archivo):
    """Línea centrada bajo una figura con un enlace al PNG en alta resolución (ruta relativa al .docx)."""
    p = OxmlElement("w:p")
    despues_de.addnext(p)
    par = Paragraph(p, despues_de.getparent())
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h = OxmlElement("w:hyperlink")
    h.set(qn("r:id"), doc.part.relate_to(f"diagramas/{archivo}", RT.HYPERLINK, is_external=True))
    r, rpr, t = OxmlElement("w:r"), OxmlElement("w:rPr"), OxmlElement("w:t")
    for tag, val in (("w:color", "2E75B6"), ("w:sz", "18"), ("w:u", "single")):
        e = OxmlElement(tag)
        e.set(qn("w:val"), val)
        rpr.append(e)
    t.text = "Abrir el diagrama en tamaño completo"
    r.extend([rpr, t])
    h.append(r)
    p.append(h)
    return p


def mover_antes(desde, hasta_excluido, destino):
    """Mueve los elementos desde `desde` hasta antes de `hasta_excluido`, delante de `destino`."""
    el = desde._p
    while el is not hasta_excluido._p:
        siguiente = el.getnext()
        destino._p.addprevious(el)
        el = siguiente


def _sin_acentos(texto):
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if not unicodedata.combining(c))


def _fila_como(modelo_tr, textos, tabla):
    """Copia una fila (conserva su formato) y le pone los textos dados."""
    tr = copy.deepcopy(modelo_tr)
    for tc, texto in zip(tr.findall(qn("w:tc")), textos):
        par = _Cell(tc, tabla).paragraphs[0]
        par.runs[0].text = texto
        for r in par.runs[1:]:
            r._r.getparent().remove(r._r)
    return tr


def agregar_glosario(doc, entradas):
    """Inserta términos en el Anexo B respetando el orden alfabético."""
    tabla = next(t for t in doc.tables if t.cell(0, 0).text.strip() == "Término")
    for termino, definicion in entradas:
        filas = tabla.rows[1:]
        nueva = _fila_como(filas[0]._tr, (termino, definicion), tabla)
        sig = next((f for f in filas if _sin_acentos(f.cells[0].text) > _sin_acentos(termino)), None)
        (sig._tr.addprevious if sig is not None else filas[-1]._tr.addnext)(nueva)


def agregar_formulas(doc, filas_nuevas):
    """Agrega filas al final de la tabla de fórmulas de la sección 2.2."""
    tabla = next(t for t in doc.tables if t.cell(0, 0).text.strip() == "Métrica"
                 and t.cell(0, 1).text.strip() == "Fórmula")
    for textos in filas_nuevas:
        tabla.rows[-1]._tr.addnext(_fila_como(tabla.rows[-1]._tr, textos, tabla))


def figura(cur, ruta, pie):
    """Imagen centrada con su pie de figura, en la posición del cursor."""
    p = cur.p("")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(ruta), width=Cm(15))
    c = cur.p(pie)
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.runs[0].italic, c.runs[0].font.size = True, Pt(9)
    c.runs[0].font.color.rgb = RGBColor(0x59, 0x74, 0x82)


def con_tipo(titulo, tipos):
    """«Cargador. » → «Cargador (componente). »; usa el mismo tipo que el ícono del diagrama."""
    base = titulo.split(" (")[0].rstrip(". ")
    tipo = tipos.get(base) or ("registro" if base.startswith("Registros") else None)
    if not tipo:
        return titulo
    return titulo.replace(" (", f" ({tipo}, ", 1) if " (" in titulo else f"{base} ({tipo}). "


def autor(cur, *runs):
    cur.p((f"{MARCA}. ", "b"), *runs, shade=True)


def fin_de_etapa(doc, prefix):
    """Último elemento de la etapa: antes del siguiente título «N.N …» o «N. …» o del salto de página."""
    el = find(doc, prefix)._p
    while (sig := el.getnext()) is not None and sig.tag != qn("w:sectPr"):
        if sig.tag == qn("w:p"):
            p = Paragraph(sig, doc._body)
            if p.style.name.startswith("Heading") and re.match(r"^\d+(\.\d+)?\.?\s", p.text.strip()):
                break
            if any(b.get(qn("w:type")) == "page" for b in sig.iter(qn("w:br"))):
                break
        el = sig
    return el


# Apertura de cada etapa: qué se hace, de qué parte y con qué termina (el documento se explica solo)
QUE_SE_HACE = {
    "1.1 Etapa 1": (
        "Se preparan las imágenes de los dos peces nuevos, Trucha y Rohu, en el mismo formato que "
        "Mendeley: solo el ojo y en dos clases, fresco y no fresco. En Trucha se revisa además la calidad "
        "de cada recorte y se quitan los que no muestran bien el ojo. Termina con 1,150 imágenes de "
        "Trucha y 1,261 de Rohu."),
    "1.2 Etapa 2": (
        "Se prepara cada experimento. Las imágenes se reparten en 10 folds: en cada corrida, 9 se usan "
        "para entrenar y 1 para evaluar (1.2.1). El modelo parte de lo que aprendió en Mendeley: esa parte "
        "se mantiene fija y solo se entrena la cabeza, la última capa, que decide si la imagen es fresco o "
        "no fresco (1.2.2)."),
    "1.3 Etapa 3": (
        "La cabeza del modelo aprende con las imágenes del pez nuevo. Después de cada época se mide el F1 "
        "macro en el fold de evaluación y se conserva la mejor época de cada fold."),
    "1.4 Etapa 4": (
        "Se mide, en cada fold, cuántas imágenes acierta y falla el modelo en cada clase (matriz de "
        "confusión) y qué tan bien separa fresco de no fresco (curva ROC y AUC)."),
    "1.5 Etapa 5": (
        "Se juntan los resultados de los 10 folds para tener un solo resultado por modelo y por pez. Como "
        "cada imagen se evaluó una sola vez, el resultado cubre todas las imágenes."),
    "1.6 Etapa 6": (
        "Se compara qué tan bien funciona cada modelo con los peces nuevos frente a lo que logró en "
        "Mendeley, con tres indicadores (2.6): TRD, qué porcentaje del desempeño de Mendeley conserva; "
        "Gap, cuánto pierde; y RBE, cuánto menos pierde ViT-Base/16 que DenseNet-121. Además, la prueba "
        "de Wilcoxon comprueba que la ventaja de ViT-Base/16 no se debe al azar."),
}

# Nota interna del autor (resumen informal; no reemplaza la explicación del documento)
ETAPAS = {
    "1.1 Etapa 1": "Se reúnen fotos de ojos de Trucha (Perú) y de Rohu (India), dos especies que los programas "
                   "no vieron en el experimento Mendeley. En Trucha, las fotos de los días 1 y 3 de "
                   "almacenamiento se consideran frescas y las de los días 5, 7 y 9, no frescas. Las fotos en "
                   "las que el ojo no se detectó con seguridad se revisaron a mano, y se descartaron las que "
                   "no lo mostraban bien. Quedan 1,150 fotos de Trucha y 1,261 de Rohu.",
    "1.2 Etapa 2": "Las fotos de Trucha y de Rohu se reparten en 10 lotes, cada uno con la misma proporción de "
                   "fresco y no fresco. En Rohu hay fotos casi idénticas tomadas en ráfaga; se colocan en el "
                   "mismo lote para que el programa no se evalúe con una foto casi igual a otra que ya usó para "
                   "aprender. También se cargan los dos programas elegidos en el experimento Mendeley.",
    "1.3 Etapa 3": "El programa conserva lo que aprendió en Mendeley y solo se ajusta su última parte, la que "
                   "decide entre fresco y no fresco, con las fotos de Trucha o de Rohu. Aprende con 9 lotes, se "
                   "mide con el décimo y repite el proceso hasta 30 veces; si pasan 8 repeticiones sin mejora, "
                   "se detiene.",
    "1.4 Etapa 4": "En cada ronda se toman las respuestas del momento en que el programa obtuvo su mejor "
                   "resultado con el lote que no usó para aprender. Se cuentan las fotos frescas y no frescas "
                   "que clasificó bien y las que clasificó mal, para ver en qué tipo de foto se equivoca más.",
    "1.5 Etapa 5": "Se juntan los 10 lotes de Trucha y los 10 de Rohu. Como cada foto se evaluó una sola vez, se "
                   "obtiene una respuesta para cada foto y se calcula el puntaje final de cada programa en cada "
                   "especie.",
    "1.6 Etapa 6": "Se compara el puntaje en Trucha y en Rohu con el obtenido en Mendeley, para medir cuánto se "
                   "pierde al cambiar de especie. ViT-Base/16 conservó más que DenseNet-121 en Trucha y en "
                   "Rohu, y una prueba estadística confirma que esa diferencia no se debe al azar.",
}

# Frases de limitación que se quitan de párrafos existentes (pedido del autor)
FRASES_LIMITACION = [
    (" Los scripts originales de preprocesamiento no están en la copia revisada, por lo que sus "
     "detalles se citan como procedimiento reportado.", ""),
    (" No se ha identificado un umbral numérico verificable.", ""),
    (" El código no explica la diferencia entre 76 y 70.", ""),
    (" No se dispone aquí del motivo individual para cada exclusión.", ""),
    ("; no se realiza una segunda evaluación sobre un conjunto independiente.", "."),
    ("Preparación reportada.", "Preparación de las imágenes."),
    ("Revisión manual reportada.", "Revisión manual."),
]

# Cajas de 6_documentacion/diagramas/arquitectura_generalizacion_7_etapas.png (generar_diagrama_7_etapas.py), en orden
ARQUITECTURA = [
    ("Selección de trucha (antes del experimento). ",
     "Del repositorio OneDrive (Perú) se toman las fotos por día de almacenamiento: los días 1 y 3 son "
     "fresco y los días 5, 7 y 9, no fresco; en total 1,196 fotos (ver 1.1)."),
    ("Selección de Rohu (antes del experimento). ",
     "Del repositorio Kaggle (India) se toman las 1,652 fotos de ojos; tras la depuración quedan 1,521."),
    ("Revisión de trucha. ",
     "Se revisaron a mano 128 recortes dudosos: se rescataron 82 y se excluyeron 46, así que de 1,196 "
     "fotos quedan 1,150."),
    ("Fotos listas. ",
     "Trucha (1,150) y Rohu (1,261), cada foto etiquetada como fresco o no fresco."),
    ("Modelo de Mendeley. ",
     "Viene del experimento Mendeley (flecha superior): la mejor versión de ViT-Base/16 o DenseNet-121. "
     "Mendeley tiene 5 checkpoints por modelo, uno por fold; el fold k usa el checkpoint ((k − 1) mod 5) + 1, "
     "así que cada checkpoint se usa en dos folds."),
    ("Dividir en 10 folds. ",
     "Arma los 10 folds una sola vez. Después, en cada vuelta, 1 fold va a «Probar en el fold», donde "
     "sirve de evaluación, y los otros 9 van a «Entrenar la cabeza». En Rohu, las "
     "fotos casi iguales tomadas en ráfaga quedan siempre en el mismo fold."),
    ("Entrenar la cabeza. ",
     "Congela el modelo y entrena solo la última capa (la cabeza) con una pasada (época) por las fotos de "
     "los 9 folds, con variaciones al azar (recorte, volteo, color, desenfoque, rotación y borrado). Cada "
     "pasada continúa desde los pesos que dejó la anterior, sin reiniciar: la cabeza aprende de a poco."),
    ("Probar en el fold. ",
     "La cabeza recién entrenada clasifica las fotos del fold de evaluación, que no usó para aprender, y "
     "se calcula el F1 de esa época. Es el mismo fold en todas las épocas de esa vuelta."),
    ("Guardar y decidir. ",
     "Compara ese F1 con el mejor anterior: si mejora, guarda una copia de la cabeza de esa época. Luego "
     "decide: si todavía no se "
     "cumple la parada (8 épocas sin mejora o 30 épocas), se repiten los tres pasos con otra pasada por los "
     "mismos folds, sin reiniciar; si se cumple, la mejor época pasa a «Evaluar el fold» (flecha «al terminar: "
     "mejor época»)."),
    ("Evaluar el fold. ",
     "Con la cabeza de la mejor época clasifica el fold, cuenta aciertos y errores por clase (matriz de "
     "confusión) y calcula la curva ROC y el AUC del fold."),
    ("Reunir 10 folds. ",
     "Junta el resultado de los 10 folds, así que cada foto se evalúa una sola vez, y calcula el F1 y el "
     "AUC de cada modelo en cada pez. El F1 va a «Comparar» y los datos, a «Gráficas»."),
    ("Gráficas. ",
     "Dibuja las matrices de confusión y las curvas ROC."),
    ("Comparar. ",
     "Recibe el F1 de cada modelo en el pez nuevo y el F1 que logró en Mendeley (flecha superior: 96.96 "
     "y 95.66). Calcula cuánto conserva (TRD), cuánto pierde (Gap) y cuánto menos pierde ViT-Base/16 que "
     "DenseNet-121 (RBE); la prueba de Wilcoxon comprueba que la diferencia no se deba al azar. Pasa los "
     "indicadores a «Documentación»."),
    ("Documentación. ",
     "Este documento: reúne las tablas, las gráficas, los indicadores y los registros."),
    ("Registros y resultados guardados. ",
     "Lo que se anota en cada época y los resultados de cada fold de la corrida de 10 folds (flechas "
     "punteadas). Respaldan cada número del documento."),
]

# Anexo A — (dato, sección, archivo en este repositorio, script que lo genera)
ORIGEN_DATOS = [
    ("Dato", "Sección", "Archivo", "Script"),
    ("Homologación de estados (3 → 2)", "1.1", "Plan de tesis, Tabla 6", "dataset_freshness.py (DAY_TO_LABEL_2)"),
    ("Clase de cada imagen de Rohu (Fresh_Eyes / Nonfresh_Eyes)", "1.1", "—", "domains.py (collect_rohu_eyes)"),
    ("Grupos de fotos casi idénticas (Rohu)", "1.2.1", "—", "detectar_duplicados.py"),
    ("F1, épocas y tiempo por fold; resultado pooled", "1.6 y 3.x",
     "3_resultados/{trucha,rohu}/linear_probe_10fold/resumen.csv",
     "linear_probe_con_test.py (--protocolo original --folds 10)"),
    ("val_F1 por época, predicciones, matrices de confusión y curvas ROC (por fold y pooled)", "1.3 a 1.5 y 3.x",
     "3_resultados/{trucha,rohu}/linear_probe_10fold/{modelo}/*.csv", "linear_probe_con_test.py"),
    ("Registro de la corrida", "1.3 a 1.6", "2_logs/linear_probe_10fold.log", "linear_probe_con_test.py"),
    ("Conjunto de Rohu (ver Nota sobre Rohu)", "Nota sobre Rohu", "2_logs/linear_probe_10fold.log",
     "experimento_dataset_mixto.py (construir_dataset_mixto)"),
    ("F1 de Mendeley (dominio fuente)", "1.6 y 2.6",
     "Mendeley_ViT_DenseNet/3_resultados_test_por_modelo/metricas_pooled_por_modelo.csv",
     "dnc_5fold_confmat_roc.py"),
    ("Prueba de Wilcoxon (ViT-Base/16 frente a DenseNet-121)", "1.6",
     "3_resultados/wilcoxon_vit_vs_densenet.csv", "wilcoxon_vit_vs_densenet.py"),
    ("TRD, Gap y RBE", "1.6 y 2.6", "Se calculan con la media de los 10 folds y el F1 de Mendeley", "metrics.py"),
    ("Gráficas por fold y pooled", "3.x", "4_graficas/{trucha,rohu}/{modelo}/",
     "generar_documentacion_tecnica.py"),
    ("Figuras de arquitectura y vista simple", "Introducción y 1",
     "6_documentacion/diagramas/ (SVG y PNG)",
     "utilitarios/diagramas/generar_diagrama_7_etapas.py"),
    ("Fórmulas de las métricas", "2", "—", "metrics.py"),
]

GLOSARIO_NUEVO = [
    ("Backpropagation", "Cálculo del gradiente del error hacia atrás por la red; con él el optimizador "
     "actualiza los pesos."),
    ("CLAHE", "Ecualización adaptativa del histograma con límite de contraste; normaliza brillo y "
     "contraste del recorte del ojo."),
    ("CosineAnnealingLR", "Ver Scheduler coseno."),
    ("CSV", "Archivo de texto con datos en columnas separadas por comas; aquí guarda métricas, matrices y "
     "curvas ROC."),
    ("CUDA", "Plataforma de NVIDIA para ejecutar cálculos en la GPU."),
    ("Detección dudosa", "Recorte del ojo cuya detección automática quedó marcada como «dudosa»; se "
     "revisa a mano y se rescata o se excluye."),
    ("Época", "Ver Epoch."),
    ("Estados de frescura", "Categorías originales de cada repositorio (muy fresco, fresco y no fresco); "
     "para este estudio se homologan a dos clases: fresco y no fresco."),
    ("Gradiente", "Dirección y magnitud en que cambia el error al variar cada peso; guía la actualización "
     "de los pesos."),
    ("Homologación de estados", "Paso de los estados de frescura de cada repositorio a las dos clases del "
     "estudio (en trucha, por día de almacenamiento)."),
    ("Logs (registros de ejecución)", "Archivos de texto donde el entrenamiento y la evaluación anotan sus "
     "resultados por época y por fold."),
    ("Optimizador", "Algoritmo que actualiza los pesos del modelo a partir del gradiente; aquí AdamW."),
    ("PNG", "Formato de imagen sin pérdida usado para las gráficas."),
    ("Recorte del ojo", "Imagen cuadrada centrada en el ojo del pez, obtenida de la foto de la cabeza "
     "completa; es la entrada del modelo."),
    ("Redimensionado (resize)", "Cambio del tamaño de la imagen al que espera el modelo (224 o 256 px)."),
    ("Rescate manual", "Revisión visual de las detecciones dudosas para recuperar las que sí muestran el "
     "ojo."),
    ("stdev", "Desviación estándar muestral (n − 1)."),
    ("EfficientNet", "Familia de CNN que escala de forma equilibrada profundidad, ancho y resolución; "
     "evaluada en Mendeley (EfficientNet-B0 y EfficientNet-Lite0)."),
    ("MobileNet", "Familia de CNN livianas pensadas para dispositivos móviles; evaluada en Mendeley "
     "(MobileNetV1, V2 y V3-Small)."),
    ("Ráfaga", "Serie de fotos tomadas una tras otra en pocos segundos; produce fotos casi idénticas."),
    ("ResNet", "Familia de CNN con conexiones residuales que saltan capas; evaluada en Mendeley (ResNet-50)."),
    ("Sobreajuste (overfitting)", "Cuando el modelo memoriza las fotos de entrenamiento en lugar de "
     "aprender patrones generales, y rinde peor con fotos nuevas."),
    ("p-valor", "Probabilidad de observar una diferencia como la medida si en realidad los dos modelos "
     "rindieran igual; si es menor que 0.05, la diferencia se considera significativa."),
    ("Prueba de Wilcoxon", "Prueba estadística no paramétrica que compara dos modelos medidos en los "
     "mismos folds, sin suponer que los datos siguen una distribución normal."),
    ("Significancia estadística (α)", "Umbral para decidir si una diferencia es real; el plan de tesis "
     "fija α = 0.05."),
    ("Swin", "Transformer de visión que calcula la atención en ventanas locales desplazadas; evaluado en "
     "Mendeley (Swin-Base)."),
    ("F1 medio (10 folds)", "Promedio simple del F1 macro de los 10 folds de un modelo en un dominio; es la "
     "cifra que usan TRD, Gap y RBE."),
]

# ViT-Base/16, Trucha — 3_resultados/trucha/linear_probe_10fold/vit_base_patch16_224/confusion_matrix_fold{k}.csv
# y resumen.csv (F1 macro de cada fold)
VIT_TRUCHA = [
    (1, 37, 12, 5, 61, "84.54"), (2, 43, 5, 7, 60, "89.33"), (3, 38, 10, 8, 59, "83.81"),
    (4, 31, 17, 6, 61, "78.54"), (5, 45, 3, 2, 65, "95.52"), (6, 40, 8, 5, 62, "88.27"),
    (7, 39, 9, 3, 64, "89.05"), (8, 39, 9, 4, 63, "88.18"), (9, 31, 17, 2, 65, "81.90"),
    (10, 34, 14, 3, 64, "84.14"),
]


INTRO_ORGANIZACION = [
    ("Sección", "Qué contiene"),
    ("Introducción", "De qué trata el documento, el resultado principal y los conceptos clave"),
    ("1. Metodología", "Las 6 etapas del experimento, desde los datos hasta la comparación"),
    ("2. Glosario de fórmulas", "Cómo se calcula cada métrica e indicador (TRD, Gap, RBE), con ejemplos"),
    ("3. Detalle por modelo", "Los números de ViT-Base/16 y DenseNet-121 en Trucha y Rohu"),
    ("4. Conclusiones", "Qué se encontró"),
    ("Anexo A", "De dónde sale cada dato"),
    ("Anexo B", "Glosario de términos técnicos"),
    ("Nota sobre Rohu", "Composición de las imágenes de Rohu"),
]

CONCEPTOS = [
    ("Concepto", "En palabras simples", "Más detalle"),
    ("Dominio fuente y dominio objetivo", "El dominio fuente es donde el modelo aprendió (Mendeley, peces de "
     "Indonesia). Los dominios objetivo son donde se lo pone a prueba: Trucha (Perú) y Rohu (India).", "1.1"),
    ("Linear probe (backbone congelado)", "Se deja fijo todo lo que el modelo aprendió en Mendeley y solo "
     "se vuelve a entrenar la última capa (la cabeza), la que decide si el ojo es fresco o no fresco.",
     "1.2.2"),
    ("Fold y fold retenido", "Las fotos se dividen en 10 partes (folds). En cada repetición, 9 partes "
     "entrenan la cabeza y la parte restante (el fold retenido) sirve de examen. Así todas las fotos se "
     "evalúan una vez.", "1.2.1"),
    ("Época (epoch) y paciencia", "Una época es una vuelta completa por las fotos de entrenamiento. Si el "
     "resultado no mejora durante 8 épocas seguidas, el entrenamiento se detiene.", "1.3"),
    ("Matriz de confusión", "Tabla que cuenta aciertos y errores para fresco y no fresco.", "2.1"),
    ("F1 macro", "Nota de 0 a 100 que resume los aciertos de las dos clases con el mismo peso. Es la "
     "métrica principal.", "2.2"),
    ("Accuracy", "Porcentaje de fotos bien clasificadas, contando las dos clases juntas.", "2.2"),
    ("Precisión y recall", "Precisión: de las fotos que el modelo llamó fresco, cuántas lo eran. Recall: de las "
     "fotos que eran frescas, cuántas encontró. Lo mismo se calcula para no fresco.", "2.2"),
    ("AUC", "Nota de 0 a 1 que indica qué tan bien el modelo separa fresco de no fresco usando sus "
     "probabilidades; 1 es perfecto.", "2.4"),
    ("Resultado pooled", "Se juntan las respuestas de los 10 folds y se calcula una sola nota general.",
     "2.5"),
    ("TRD", "Qué porcentaje de su nota en Mendeley conserva el modelo en el nuevo dominio; 100 % significa "
     "que no pierde nada.", "2.6"),
    ("Gap", "Cuánto aumenta el error al pasar de Mendeley al nuevo dominio; mientras más bajo, mejor.", "2.6"),
    ("RBE", "Cuánto reduce ViT-Base/16 la brecha de error en comparación con DenseNet-121; si es positivo, "
     "ViT-Base/16 se adapta mejor.", "2.6"),
    ("Prueba de Wilcoxon", "Prueba estadística que indica si la ventaja de un modelo sobre otro es real "
     "o podría deberse al azar. Si p es menor que 0.05, la diferencia es significativa.", "1.6"),
    ("F1 medio y F1 pooled", "De la misma corrida de 10 folds salen dos resúmenes: el F1 medio, promedio del "
     "F1 de los 10 folds, que usan TRD, Gap y RBE; y el F1 pooled, calculado una vez con todas las respuestas "
     "juntas. Difieren poco (por ejemplo, 86.33 frente a 86.42 en ViT-Base/16, Trucha).", "1.6"),
]


def para_publico_general(doc):
    """Agrega lo que necesita alguien que no es del área para seguir el documento de principio a fin."""
    # Introducción antes de la metodología
    metodologia = find(doc, "1. Metodología")
    cur = Cursor(doc, metodologia)
    cur.el = metodologia._p.getprevious()
    cur.p("Introducción", style="Heading 1")
    cur.p(("¿De qué trata este documento? ", "b"),
          "En el primer experimento (documento de Mendeley), ViT-Base/16 y DenseNet-121 aprendieron a "
          "reconocer ojos de pescado frescos y no frescos con fotos de peces de Indonesia. Aquí se prueba si "
          "ese aprendizaje sirve con otros peces y otras fotos: trucha de Perú y Rohu de India. A esto se le "
          "llama generalización entre conjuntos de datos (cross-dataset). A cada modelo solo se le vuelve a "
          "entrenar la última capa, y se mide cuánto rinde frente a lo que lograba en Mendeley. Las fotos de "
          "trucha y Rohu no se usaron en ningún momento del entrenamiento en Mendeley; allí solo se eligieron "
          "los dos modelos.")
    cur.p(("Objetivo de la tesis que responde. ", "b"),
          "Este documento responde al objetivo específico 2 del plan de tesis: «Determinar el impacto de la "
          "variabilidad visual entre distintos conjuntos de datos sobre el desempeño de los modelos de "
          "Machine Learning identificados». Su producto es la evidencia cuantitativa del efecto de la "
          "variabilidad visual sobre el desempeño de los modelos en escenarios cross-dataset (plan de "
          "tesis, Tabla 9). Los objetivos específicos 3 y 4 se desarrollan en el documento de adaptación "
          "de dominio (CORAL).")
    cur.p(("Resultado principal. ", "b"),
          "ViT-Base/16 conserva más de su desempeño que DenseNet-121 en los dos dominios. En Trucha, "
          "ViT-Base/16 conserva el 89.04 % de su F1 de Mendeley (baja de 96.96 a 86.33) y DenseNet-121 el "
          "86.53 % (de 95.66 a 82.77). Dicho de otra forma, DenseNet-121 pierde 12.89 puntos y ViT-Base/16 "
          "10.63: ViT-Base/16 pierde un 17.53 % menos. En Rohu, ViT-Base/16 conserva el 93.01 % y "
          "DenseNet-121 el 87.98 %, y ViT-Base/16 pierde un 41.04 % menos. Los resultados salen de una "
          "validación cruzada de 10 folds. Esa ventaja es "
          "estadísticamente significativa en los dos dominios (prueba de Wilcoxon).")
    figura(cur, FIGURA.with_name("vista_simple_generalizacion.png"),
           "La generalización en 6 pasos. El detalle de cada etapa está en la sección 1.")
    cur.p("Por qué esta etapa se diseñó así", style="Heading 2")
    cur.p("Esta etapa mide cuánto de lo aprendido en Mendeley sirve con fotos de otros peces. Las "
          "decisiones del diseño se derivan de ese propósito:")
    for titulo, texto in [
        ("Solo se reentrena la última capa. ",
         "Si se reentrenara todo el modelo, lo aprendido en Mendeley se mezclaría con un aprendizaje nuevo y "
         "no se podría medir cuánto sirve por sí solo (ver 1.2.2)."),
        ("10 folds y evaluación en el fold retenido. ",
         "Con 1,150 y 1,261 imágenes, separar además un test aparte dejaría muy pocas imágenes para entrenar "
         "y para evaluar; así, cada imagen se evalúa una vez (ver 1.2.1). Las primeras pruebas se hicieron "
         "con 5 folds; la versión final usa 10 folds por recomendación del asesor: cada fold entrena con más "
         "imágenes (90 %) y hay 10 resultados por modelo para la prueba de Wilcoxon."),
        ("Mismas condiciones para los dos modelos. ",
         "ViT-Base/16 y DenseNet-121 usan los mismos folds, la misma semilla y la misma configuración, para "
         "que las diferencias se deban al modelo."),
        ("Comparación con Mendeley. ",
         "TRD, Gap y RBE comparan el resultado en cada pez con el de Mendeley, para medir cuánto se pierde "
         "al cambiar de fotos (ver 2.6)."),
        ("Prueba de Wilcoxon. ",
         "Comprueba que la ventaja de ViT-Base/16 sobre DenseNet-121 no se deba al azar, comparando los 10 "
         "folds; con 5 no podría llegar a 0.05 (ver 1.6)."),
    ]:
        cur.p((titulo, "b"), texto, style="List Bullet")
    cur.p("Cómo está organizado el documento", style="Heading 2")
    cur.table(INTRO_ORGANIZACION)
    cur.p("Conceptos clave", style="Heading 2")
    cur.p("Estos son los términos que más se repiten. La última columna indica la sección donde se "
          "explican en detalle; el resto de términos está en el glosario del Anexo B.", shade=False)
    cur.table(CONCEPTOS)
    cur.p("Una sola corrida de 10 folds", style="Heading 3")
    cur.p("Todas las cifras de este documento (tablas, matrices, curvas ROC, F1 por época, TRD, Gap, RBE y "
          "prueba de Wilcoxon) salen de una sola corrida: linear_probe_con_test.py --protocolo original "
          "--folds 10. Cada imagen se evalúa exactamente una vez, en el fold que queda fuera: en Trucha, 10 "
          "folds de 115 imágenes; en Rohu, folds de 117 a 133. La suma de las imágenes de evaluación de los "
          "10 folds (3_resultados/{trucha,rohu}/linear_probe_10fold/resumen.csv) y la de cada matriz de "
          "confusión pooled dan 1,150 en Trucha y 1,261 en Rohu.")


    # Frase «en palabras simples» antes de los hiperparámetros
    ref = find(doc, "Hiperparámetros (linear_probe_con_test.py")
    cur = Cursor(doc, ref)
    cur.el = ref._p.getprevious()
    cur.p("En palabras simples: los dos modelos se entrenan con la misma receta, para que la comparación "
          "sea justa. Los hiperparámetros son los ajustes de esa receta (qué tan rápido aprende la cabeza, "
          "cuántas épocas como máximo, cuándo detenerse).")

    # Justificaciones (borradores para que el autor las revise)
    cur = Cursor(doc, find(doc, "Cada fold parte de un checkpoint de Mendeley"))
    cur.p(("Por qué se reutilizan los 5 checkpoints de Mendeley. ", "b"),
          "Mendeley se entrenó con 5 folds, así que cada modelo tiene 5 checkpoints. Con 10 folds, el fold k "
          "parte del checkpoint ((k − 1) mod 5) + 1: los folds 1 y 6 usan el checkpoint 1, los folds 2 y 7 el "
          "checkpoint 2, y así hasta los folds 5 y 10. Cada checkpoint se usa dos veces, de modo que el "
          "resultado no depende de un solo modelo de Mendeley, y ninguno vio fotos de Trucha ni de Rohu.")
    cur.p(("Por qué solo se entrena la cabeza. ", "b"),
          "El objetivo de esta etapa es medir cuánto sirve en otro dominio lo que el modelo aprendió en "
          "Mendeley. Si se reentrenara todo el modelo, ese efecto se mezclaría con un aprendizaje nuevo y no "
          "se podría saber cuánto se debe a cada uno. Además, con 1,035 a 1,144 imágenes de entrenamiento por "
          "fold, reentrenar los 85.8 millones de parámetros de ViT-Base/16 llevaría a que el modelo memorice "
          "las fotos (sobreajuste); entrenar solo la cabeza (1,538 parámetros en ViT-Base/16 y 2,050 en "
          "DenseNet-121) evita ese riesgo. Por último, este resultado es el punto de comparación para la "
          "adaptación de dominio, la siguiente etapa de la tesis.")
    cur = Cursor(doc, find(doc, "Variaciones al azar en entrenamiento"))
    cur.p(("Por qué estos hiperparámetros. ", "b"),
          "ViT-Base/16 y DenseNet-121 usan la misma configuración en Trucha y en Rohu, para que las "
          "diferencias se deban al modelo y al dominio, y no al ajuste. El learning rate de 1e-4 con AdamW "
          "es el mismo que en Mendeley. El weight decay (1e-4), la función de pérdida (CrossEntropyLoss) y "
          "la ausencia de warmup difieren de Mendeley (0.05, Focal Loss y 3 épocas de warmup): en pruebas "
          "previas con el backbone congelado, esta configuración dio los mejores resultados. Además, "
          "el límite de 12 épocas del ajuste completo cortaba la cabeza antes de su mejor "
          "resultado; se amplió a 30. En esta corrida, 35 de los 40 folds se detuvieron por paciencia "
          "antes del límite.")
    cur = Cursor(doc, find(doc, "Por qué algunos folds terminan antes de 30 epochs"))
    cur.p(("Por qué la paciencia es 8 y no 7 como en Mendeley. ", "b"),
          "En Mendeley la época se elige con 960 imágenes de validación por fold; en Trucha y Rohu, con el "
          "fold retenido, que tiene entre 115 y 133 imágenes. Con tan pocas imágenes, cada foto mal "
          "clasificada mueve el accuracy casi un punto (1/115 ≈ 0.87 puntos), así que el F1 sube y baja "
          "más de una época a otra. Una época más de paciencia evita detener el entrenamiento por una caída "
          "pasajera.")

    # Prueba de Wilcoxon (3_resultados/wilcoxon_vit_vs_densenet.csv)
    doc_res = find(doc, "Documentación de resultados")
    cur = Cursor(doc, doc_res)
    cur.el = doc_res._p.getprevious()
    cur.p("Prueba de Wilcoxon: ViT-Base/16 frente a DenseNet-121", style="Heading 3")
    cur.p("En palabras simples: la prueba indica si la ventaja de ViT-Base/16 sobre DenseNet-121 es real o "
          "podría deberse al azar. Compara los dos modelos fold por fold; si p es menor que 0.05 (el nivel "
          "α que fija el plan de tesis), la diferencia es estadísticamente significativa.")
    cur.p("Se comparan los 10 folds de cada dominio (los mismos folds para los dos modelos). Con 10 pares de "
          "resultados la prueba puede llegar a 0.05; con 5 folds el p-valor más bajo posible sería 0.0625.",
          shade=False)
    cur.table([("Dominio", "Folds en que gana ViT-Base/16", "F1 medio ViT-Base/16", "F1 medio DenseNet-121",
                "W", "p-valor", "¿Significativo?"),
               ("Trucha", "8 de 10", "86.33", "82.77", "8", "0.0488", "Sí"),
               ("Rohu", "9 de 10", "90.18", "84.16", "2", "0.0059", "Sí")])
    cur.p("Fuente: 3_resultados/wilcoxon_vit_vs_densenet.csv, generado con "
          "1_codigo/wilcoxon_vit_vs_densenet.py.", shade=False)

    # Guía de lectura de la sección 3
    cur = Cursor(doc, find(doc, "3. Detalle por modelo"))
    cur.p(("Cómo leer esta sección. ", "b"),
          "Cada modelo tiene siete partes: x.1 muestra el entrenamiento época por época; x.2 a x.4 son los "
          "resultados en Trucha (nota de cada fold con su matriz, curvas ROC y nota general pooled); y x.5 a "
          "x.7 repiten lo mismo para Rohu. La nota general pooled de cada dominio es la que aparece en la "
          "tabla de 1.6. Los modelos son ViT-Base/16 y DenseNet-121: ver 3.1 y 3.2.")

    # Conclusiones antes de los anexos
    anexo_a = find(doc, "Anexo A.")
    cur = Cursor(doc, anexo_a)
    cur.el = anexo_a._p.getprevious()
    cur.p("4. Conclusiones", style="Heading 1")
    for texto in [
        "Los dos modelos bajan al pasar de Mendeley a los nuevos dominios: ViT-Base/16 pasa de 96.96 % de "
        "F1 macro a 86.33 % en Trucha y 90.18 % en Rohu; DenseNet-121 pasa de 95.66 % a 82.77 % y 84.16 % "
        "(media de 10 folds).",
        "ViT-Base/16 retiene más desempeño (TRD de 89.04 % y 93.01 %, frente a 86.53 % y 87.98 % de "
        "DenseNet-121) y su brecha de error es menor (Gap de 10.63 y 6.78 puntos, frente a 12.89 y 11.50).",
        "El RBE es positivo en los dos dominios: ViT-Base/16 pierde un 17.53 % menos que DenseNet-121 en "
        "Trucha y un 41.04 % menos en Rohu, es decir, se adapta mejor que la CNN de referencia.",
        "La ventaja de ViT-Base/16 es estadísticamente significativa en los dos dominios (Wilcoxon: "
        "p = 0.0488 en Trucha y p = 0.0059 en Rohu).",
        "Este resultado respalda usar ViT-Base/16 como base del modelo con adaptación de dominio, la "
        "siguiente etapa de la tesis.",
    ]:
        cur.p(texto, style="List Bullet")


def actualizar():
    doc = Document(DOC)
    if any(p.text.startswith(MARCA) for p in doc.paragraphs):
        print("Ya actualizado; no se hace nada.")
        return
    shutil.copy(DOC, DOC.with_suffix(".antes_reunion.docx"))

    # Tarea 6 — explicación propia al inicio de cada etapa
    # Figura inicial: versión regenerada con generar_diagrama_7_etapas.py
    reemplazar_imagen(doc, next(doc.element.body.iter(qn("a:blip"))), FIGURA)

    # Tarea 2 — arquitectura caja por caja (debajo de la figura inicial)
    pie = find(doc, "Arquitectura del sistema de generalización")
    pie.runs[0].text = "Arquitectura del sistema de generalización: selección de imágenes y seis etapas."
    for r in pie.runs[1:]:
        r._r.getparent().remove(r._r)
    cur = Cursor(doc, pie)
    # Título propio para la arquitectura y, bajo el pie, un enlace a la imagen en alta resolución
    titulo = OxmlElement("w:p")
    pie._p.getprevious().addprevious(titulo)  # antes de la imagen
    Paragraph(titulo, pie._parent).add_run("Arquitectura del sistema")
    Paragraph(titulo, pie._parent).style = doc.styles["Heading 2"]
    cur.el = enlace_ampliado(doc, pie._p, FIGURA.name)  # la explicación sigue después del enlace
    cur.p("Explicación de la arquitectura, componente por componente", style="Heading 3")
    cur.p("El diagrama se lee de izquierda a derecha: primero la selección de las imágenes, "
          "después las seis etapas. La flecha continua es paso de datos y "
          "la punteada, un registro que se guarda o se consulta. El ícono de cada caja indica su tipo: "
          "componente (programa que hace una tarea), función (paso de cálculo), dataset (conjunto "
          "de fotos), archivo (resultado guardado) o registro (logs de ejecución).")
    for titulo, texto in ARQUITECTURA:
        cur.p((con_tipo(titulo, TIPO_CAJA), "b"), texto)

    # Tarea 3 — depuración de trucha en dos pasos
    cur = Cursor(doc, find(doc, "Depuración de imágenes de trucha"))
    cur.p(("Paso 1 — Homologación de estados (3 → 2). ", "b"),
          "Las fotos de trucha están organizadas por día de almacenamiento. Según el plan de tesis "
          "(Tabla 6), el día 1 corresponde a muy fresco (245 imágenes), el día 3 a fresco (249) y el día 5 "
          "a no fresco (240), en total 734 imágenes. El experimento agrega los días 7 y 9 (235 y 227 "
          "imágenes), que también son no fresco, y llega a 1,196. Para la clasificación binaria, igual que "
          "en Mendeley, muy fresco y fresco se unen en fresco (días 1 y 3: 494 imágenes) y los días 5, 7 y "
          "9 quedan en no fresco (702 imágenes) (DAY_TO_LABEL_2 en 1_codigo/dataset_freshness.py).")
    cur.p(("Paso 2 — Exclusión por calidad. ", "b"),
          "Una vez en dos clases, se revisó la calidad del recorte del ojo: de 1,196 imágenes se "
          "excluyeron 46 (3.8 %) y quedaron 1,150, con 481 fresco y 669 no fresco. No se aplicó "
          "aumento de datos, por lo que las clases quedan desbalanceadas; por eso la métrica central "
          "es el F1 macro. Los párrafos siguientes detallan la revisión.")

    # Tarea 5 — «Limitación» → justificación; sin párrafos de limitaciones (pedido del autor)
    lim = find(doc, "Limitación: no hay un conjunto de prueba independiente")
    lim.text = ("Justificación: con 1,150 y 1,261 imágenes, separar además un test independiente "
                "dejaría menos datos para entrenar la cabeza y folds de evaluación muy pequeños. Por "
                "eso cada imagen se evalúa una vez, en el fold retenido, y ese mismo fold se usa para "
                "elegir la mejor época. El mismo procedimiento se aplica a ViT-Base/16 y a DenseNet-121, "
                "así que la comparación entre los dos y los indicadores TRD, Gap y RBE no se ven afectados.")
    rev = find(doc, "Revisión manual")
    rev.text = rev.text.replace(
        "De los 76 restantes, se indica una tercera revisión de 70 imágenes originales, con 30 recortes "
        "recuperados y verificados.",
        "De los 76 restantes, en una tercera pasada se revisaron 70 imágenes originales y se recuperaron 30 "
        "recortes; los 46 restantes (76 − 30), incluidas las 6 que no pasaron a esa revisión, se excluyeron.")
    assert "tercera pasada" in rev.text
    quitar(find(doc, "Rohu rinde por encima de Trucha"))
    quitar(find(doc, "Alcance de la evidencia"))
    for p in doc.paragraphs:
        for vieja, nueva in FRASES_LIMITACION:
            if vieja in p.text:
                p.text = p.text.replace(vieja, nueva)

    # Tarea 8 — por qué algunos folds terminan antes de 30 epochs
    cur = Cursor(doc, find(doc, "Nota. Epochs ejecutados"))
    cur.p(("Por qué algunos folds terminan antes de 30 epochs. ", "b"),
          "Cada fold corre como máximo 30 epochs (300 por modelo y dominio). Si el F1 macro del fold "
          "retenido no mejora durante 8 epochs seguidas, el entrenamiento se detiene "
          "(linear_probe_con_test.py, entrenar_fold, líneas 147–157). Por eso, en un fold con menos de 30 "
          "epochs, la mejor época es la última época corrida menos 8. Ejemplo, ViT-Base/16 en Trucha: el fold 1 "
          "corrió 18 epochs, así que su mejor época fue la 10; el fold 3 corrió 23 y su mejor época fue la "
          "15. Siempre se reporta la mejor época, no la última.")

    # Tarea 7 — cómo se obtiene cada columna de la tabla general
    cur = Cursor(doc, find(doc, "Nota: el F1 pooled se calcula una vez"))
    cur.p("Cómo se obtiene cada columna de esta tabla", style="Heading 3")
    cur.p("Cada número se calcula una sola vez sumando los 10 folds (ViT-Base/16 en Trucha, datos de "
               "3.1.2 y 3.1.4):")
    cur.table([("Fold", "TP", "FN", "FP", "TN", "F1 macro del fold")] +
              [(f"Fold {k}", tp, fn, fp, tn, f1) for k, tp, fn, fp, tn, f1 in VIT_TRUCHA] +
              [("Suma / media", "377", "104", "45", "624", "86.33")], bold_last=True)
    cur.p("Con esa matriz sumada se obtiene cada columna de la tabla:", shade=False)
    cur.table([
        ("Columna", "Cómo se obtiene", "Cálculo (ViT-Base/16, Trucha)", "Valor"),
        ("Accuracy", "Aciertos de la matriz sumada / total", "(377 + 624) / 1,150 × 100", "87.04"),
        ("Precisión macro", "Promedio de la precisión de cada clase (matriz sumada)",
         "fresco 377 / (377 + 45) = 89.34\nno fresco 624 / (624 + 104) = 85.71\n"
         "(89.34 + 85.71) / 2", "87.53"),
        ("F1 macro (pooled)", "Promedio del F1 de cada clase (matriz sumada)",
         "(83.50 + 89.33) / 2", "86.42"),
        ("Desv. std F1", "Desviación estándar muestral (n − 1) de los 10 F1 de fold",
         "stdev de los 10 F1 de la tabla anterior", "4.75"),
        ("AUC (pooled)", "Con las 1,150 probabilidades concatenadas (roc_curve_pooled.csv), no con "
                         "la matriz", "Área bajo la ROC pooled", "0.9224"),
        ("F1 medio (10 folds)", "Media simple de los 10 F1 de fold; es la cifra que usan TRD, Gap y RBE",
         "suma de los 10 F1 de la tabla anterior / 10", "86.33"),
    ])

    # Ejemplo de 1.5 con enlace a la sección de donde salen los datos
    find(doc, "Ejemplo instanciado — ViT-Base/16, Trucha").add_run(" (datos de 3.1.2 y 3.1.4)")

    # Glosario 2.6: ejemplo resuelto de TRD, Gap y RBE
    cur = Cursor(doc, find(doc, "Para el propio baseline, Gap_baseline"))
    cur.p("Ejemplo — ViT-Base/16 en Trucha (F1 medio de 10 folds de la tabla de 1.6; F1 de Mendeley de "
          "Documentacion_Tecnica_10_Modelos.docx, tabla de la etapa 6):", shade=False)
    cur.f("TRD = 86.33 / 96.96 × 100 = 89.04 %",
          "Error_objetivo = 1 − 86.33/100 = 0.1367",
          "Error_fuente = 1 − 96.96/100 = 0.0304",
          "Gap = (0.1367 − 0.0304) × 100 = 10.63 puntos",
          "Gap_DenseNet = [(1 − 0.8277) − (1 − 0.9566)] × 100 = 12.89 puntos",
          "RBE = (12.89 − 10.63) / 12.89 × 100 = 17.53 %")

    # Composición de Rohu: una sola nota al final del documento (pedido del autor)
    # Fuente: 2_logs/linear_probe_10fold.log (línea 481) y experimento_dataset_mixto.py
    cur = Cursor(doc, doc.add_paragraph("Nota sobre Rohu", style="Heading 1"))
    cur.p("Las 1,261 imágenes de Rohu evaluadas en este documento combinan 805 imágenes de trucha (70 % "
          "de Trucha, muestreo estratificado: 337 fresco y 468 no fresco) y 456 imágenes de Rohu (30 % "
          "de las imágenes de Rohu depuradas: 85 fresco y 371 no fresco). Así lo construye "
          "experimento_dataset_mixto.py (construir_dataset_mixto, fracciones 0.7/0.3) y lo registra "
          "2_logs/linear_probe_10fold.log. En la depuración de las imágenes de Rohu se excluyeron 22 "
          "detecciones dudosas y 109 falsos positivos confirmados (escama, aleta o imagen borrosa), y se "
          "rescataron 48 dudosas tras revisión manual.", shade=False)
    cur.p("Los 10 folds se arman sobre este conjunto combinado; por eso, las cifras de Rohu y su comparación "
          "con Trucha reflejan esta combinación.", shade=False)
    cur.p("La combinación se hizo para que Rohu tuviera una cantidad de imágenes similar a la de Trucha "
          "(1,261 frente a 1,150).", shade=False)
    cur.p("Las carpetas de ojos del repositorio Kaggle tienen 1,652 imágenes de Rohu; la cifra de 717 de la "
          "Tabla 7 del plan no corresponde a ninguna carpeta (contando también las imágenes de branquias, "
          "el repositorio tiene 3,926). Después de la depuración quedan 1,521 imágenes (1,652 − 22 "
          "dudosas − 109 falsos positivos), de las que se toma el 30 %: 456.", shade=False)

    # Origen de las imágenes (plan de tesis, Tabla 8; preprocesamiento en domains.py)
    trucha = find(doc, "Trucha: 1 150 imágenes")
    cur = Cursor(doc, trucha)
    cur.el = trucha._p.getprevious()
    cur.p(("Origen de las imágenes. ", "b"),
          "Trucha proviene del repositorio OneDrive (Mamani & Pedraza, 2023; Perú): fotos de la cabeza "
          "completa tomadas con un ZTE Blade V10 Vita (3120×4160 px) durante nueve días de "
          "almacenamiento. Rohu proviene del repositorio Kaggle (JIS College of Engineering, 2024; India): "
          "fotos de la cabeza completa tomadas con un iPhone 12 (3024×4032 px) durante siete días. Como "
          "Mendeley ya trae el ojo recortado, en ambos se detecta el ojo, se recorta un cuadrado centrado "
          "en él, se normaliza el contraste con CLAHE y se lleva a 330×330 px (domains.py).", shade=False)
    for p in (trucha, find(doc, "Rohu: 1 261 imágenes")):  # mismo separador de miles que el resto
        p.text = p.text.replace("1 150", "1,150").replace("1 261", "1,261")

    # Equipo y software: dentro de «Arquitectura del sistema», antes de la etapa 1
    bloque = find(doc, "Arquitectura del sistema y entorno de ejecución")
    quitar(find(doc, "La figura inicial muestra las seis etapas"))
    mover_antes(bloque, find(doc, "1.3 Etapa 3"), find(doc, "1.1 Etapa 1"))
    bloque.text = "Equipo y software usados"
    bloque.style = doc.styles["Heading 3"]
    mendeley = find(doc, "Especificaciones técnicas de Mendeley (referencia)")
    mendeley.style = doc.styles["Normal"]
    for r in mendeley.runs:
        r.bold = True

    # Público general: introducción, guía de lectura, conclusiones y frases «en palabras simples»
    para_publico_general(doc)

    # Anexo A: de dónde sale cada dato
    intro = find(doc, "Todos los números de este documento provienen")
    viejo = intro._p.getnext()
    intro.text = ("Todos los números de este documento provienen de archivos reales. La tabla indica, para "
                  "cada dato, la sección donde aparece, el archivo del repositorio de donde sale y el "
                  "script que lo genera.")
    Cursor(doc, intro).table(ORIGEN_DATOS)
    viejo.getparent().remove(viejo)

    # «MPS» no se usa en este documento: se quita del glosario
    glosario = next(t for t in doc.tables if t.cell(0, 0).text.strip() == "Término")
    for fila in list(glosario.rows):
        if fila.cells[0].text.strip() == "MPS":
            fila._tr.getparent().remove(fila._tr)

    # Glosario: términos usados en el documento que no estaban definidos
    # Cómo se etiqueta Rohu: por carpeta del repositorio, no por día de almacenamiento
    cur = Cursor(doc, find(doc, "Rohu: 1,261 imágenes"))
    cur.p(("Clase de Rohu. ", "b"),
          "A diferencia de Trucha, la clase de Rohu no sale del día de almacenamiento, sino de la carpeta del "
          "repositorio de Kaggle: las fotos de Fresh_Eyes son fresco y las de Nonfresh_Eyes, no fresco. Se "
          "toman las carpetas de Training/ y de Testing/ juntas; las carpetas de branquias (*_Gills) se "
          "descartan, porque solo se leen las que terminan en _Eyes.")
    cur.p("Fuente: collect_rohu_eyes en domains.py (líneas 240 y 243).", shade=False)
    # Revisión: un solo nombre por corrida, términos unificados y aclaraciones que ya no aplican
    # Gap en puntos de F1, la misma escala del resto del documento
    for viejo, nuevo in (
            ("Gap = Error_objetivo − Error_fuente", "Gap = (Error_objetivo − Error_fuente) × 100, en puntos de F1"),
            ("Error_objetivo − Error_fuente (ver", "(Error_objetivo − Error_fuente) × 100, en puntos de F1 (ver"),
            ("0.1063", "10.63"), ("0.1289", "12.89"), ("0.0678", "6.78"), ("0.1150", "11.50")):
        assert reemplazar_texto(doc, viejo, nuevo), viejo
    indicadores = next(t for t in doc.tables if "TRD (%)" in [c.text.strip() for c in t.rows[0].cells])
    gap = next(c for c in indicadores.rows[0].cells if c.text.strip() == "Gap")
    assert reemplazar_texto_celda(gap, "Gap", "Gap (puntos)")
    # El entorno de generalización también usó GPU NVIDIA (dato del autor)
    entorno = next(t for t in doc.tables if t.cell(1, 1).text.startswith("PC con procesador"))
    entorno.rows[1]._tr.addnext(_fila_como(entorno.rows[1]._tr, ("Acelerador", "GPU NVIDIA vía CUDA — torch.cuda"),
                                           entorno))
    # Mismo equipo que Mendeley: mismo nombre y mismos datos de sistema
    assert reemplazar_texto_celda(entorno.cell(1, 1), "PC con procesador Intel Core i9 y 64 GB de RAM",
                                  "Servidor con procesador Intel Core i9, 64 GB de RAM")
    for fila in (("Intérprete Python", "Python 3.9.6"), ("Sistema operativo", "Windows Server")):
        entorno.rows[2]._tr.addnext(_fila_como(entorno.rows[2]._tr, fila, entorno))
    quitar_aclaracion_f1(doc)
    quitar_del_glosario(doc, "Data augmentation")
    agregar_glosario(doc, [
        ("Aumento de datos (data augmentation)", "Generación de copias modificadas de fotos originales (volteo, giro, "
         "brillo, color, desenfoque) para completar la clase con menos imágenes. En Mendeley lleva cada clase a "
         "4,000 imágenes; en Trucha y Rohu no se aplica."),
        ("Variaciones al azar en entrenamiento", "Cambios aleatorios (recorte, volteo, color, rotación, desenfoque, "
         "borrado) que se aplican a cada foto de entrenamiento en cada época; no crean fotos nuevas y no se "
         "aplican al fold de evaluación."),
    ])
    agregar_glosario(doc, GLOSARIO_NUEVO)
    agregar_formulas(doc, [
        ("Desv. std del F1 (n − 1)", "√( Σ (F1ᵢ − media)² / (n − 1) ), con n = 10 folds",
         "Cuánto varía el F1 entre folds; un valor bajo indica un modelo estable"),
        ("F1 medio (10 folds)", "(F1_fold1 + F1_fold2 + … + F1_fold10) / 10",
         "Media simple de los 10 folds; es la base de TRD, Gap y RBE"),
        ("Prueba de Wilcoxon pareada", "W = min(R⁺, R⁻): la menor suma de rangos de las diferencias "
         "positivas o negativas entre los dos modelos; p se obtiene de su distribución exacta",
         "Si p < 0.05, la diferencia entre los dos modelos es significativa"),
    ])

    for prefix, texto in QUE_SE_HACE.items():
        Cursor(doc, find(doc, prefix)).p(("Qué se hace en esta etapa. ", "b"), texto)

    # Un solo comentario del autor por etapa, al final de la etapa
    for prefix, texto in ETAPAS.items():
        cur = Cursor(doc, find(doc, prefix))
        cur.el = fin_de_etapa(doc, prefix)
        autor(cur, texto)

    for p in doc.paragraphs:
        for r in p.runs:  # asignar r.text borra lo que no es texto (imágenes): solo si cambia
            if ", misma configuración, reentrenada)" in r.text:
                r.text = r.text.replace(", misma configuración, reentrenada)", ", misma configuración)")
    doc.save(DOC)
    print(f"Actualizado: {DOC}")


if __name__ == "__main__":
    actualizar()
