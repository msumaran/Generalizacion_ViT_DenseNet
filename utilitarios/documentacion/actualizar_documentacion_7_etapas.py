"""Alinea la metodología y el diagrama con el flujo vigente de seis etapas."""
from pathlib import Path
from copy import deepcopy
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / '6_documentacion' / 'Documentacion_Tecnica_ViT_DenseNet.docx'
DIAGRAM = ROOT / '6_documentacion' / 'diagramas' / 'arquitectura_generalizacion_7_etapas.png'

def insert_before(ref, content, style=None):
    el = OxmlElement('w:p')
    ref._p.addprevious(el)
    p = Paragraph(el, ref._parent)
    if style: p.style = style
    p.add_run(content)
    return p

def actualizar():
    doc = Document(DOC)
    ps = doc.paragraphs
    if any(p.text == 'Documentación de resultados' for p in ps):
        return
    def find(prefix):
        return next(p for p in doc.paragraphs if p.text.startswith(prefix))
    find('El experimento mide cuánto').text = ('El experimento evalúa ViT-Base/16 y DenseNet-121 a partir de checkpoints de Mendeley, mediante entrenamiento de la cabeza en dos conjuntos objetivo: Trucha y Rohu. La evaluación utiliza diez folds; el fold retenido también se usa para seleccionar la época.')
    first = ps[8]; first.clear(); first.add_run().add_picture(str(DIAGRAM), width=Cm(16))
    find('1. Metodología — flujo en 7 etapas').text = '1. Metodología — flujo en 6 etapas'
    caption = find('Flujo metodológico —')
    caption.text = 'Arquitectura del sistema de generalización: seis etapas, fuentes, resultados y almacenamiento del experimento.'
    caption.runs[0].font.size = Pt(9)
    caption.runs[0].italic = True
    caption.runs[0].font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    find('Trucha. Trucha:').text = 'Trucha: 1 150 imágenes en el conjunto final, 481 fresco y 669 no fresco. Se recuperaron 82 imágenes mediante revisión visual y se excluyeron 46.'
    find('Rohu. Rohu:').text = ('Rohu: 1 261 imágenes, 422 fresco y 839 no fresco, evaluadas con el mismo protocolo de diez folds.')
    ref=find('Las clases no están balanceadas')
    blocks=[('Depuración de imágenes de trucha', 'Heading 3'),
      ('Preparación reportada. El comentario de 1_codigo/domains.py describe imágenes originales de la cabeza, detección del ojo, recorte, ajuste de contraste y redimensionamiento. El experimento utiliza los recortes ya procesados. Los scripts originales de preprocesamiento no están en la copia revisada, por lo que sus detalles se citan como procedimiento reportado.', None),
      ('Criterio de inclusión. collect_trucha consulta el estado de cada archivo en ojos_procesados/deteccion_scores.csv. Conserva los marcados «ok» y también los incluidos en verificacion_preprocesamiento/trucha_rescatadas.csv; excluye los demás. El código indica que la puntuación por sí sola no separaba de forma fiable los recortes dudosos recuperables de las detecciones erróneas. No se ha identificado un umbral numérico verificable.', None),
      ('Revisión manual reportada. Se revisaron 128 recortes de trucha marcados «dudosa». En las dos primeras pasadas se aceptaron 52 tras inspeccionar los recortes automáticos. De los 76 restantes, se indica una tercera revisión de 70 imágenes originales, con 30 recortes recuperados y verificados. El código no explica la diferencia entre 76 y 70.', None),
      ('Balance y criterio visual. En total se recuperaron 82 imágenes (52 + 30) y se excluyeron 46 (128 − 82). La revisión buscó confirmar un ojo y un recorte adecuadamente centrado; el código menciona detecciones de escamas, aletas o fondo como posibles errores. No se dispone aquí del motivo individual para cada exclusión.', None),
      ('Alcance de la evidencia. Las cifras anteriores proceden de los comentarios de domains.py y del documento existente. Los archivos de estados, rescates y revisión individual y las imágenes originales no se encontraron en las copias locales revisadas; por ello no se verificó el resultado imagen por imagen. Si faltan los CSV, la función usa estructuras vacías y la selección no reproduce el conjunto reportado.', None)]
    for t,style in blocks: insert_before(ref,t,style)
    replacements={'1.2 Etapa 2 — Validación cruzada':'1.2 Etapa 2 — Preparación del experimento',
    '1.3 Etapa 3 — Configuración':'1.2.2 Configuración de los modelos',
    '1.4 Etapa 4 — Entrenamiento':'1.3 Etapa 3 — Entrenamiento de la cabeza y selección de la mejor época',
    '1.5 Etapa 5 — Evaluación':'1.4 Etapa 4 — Evaluación',
    '1.6 Etapa 6 — Consolidación':'1.5 Etapa 5 — Reunir los resultados de los folds',
    '1.7 Etapa 7 — Comparación':'1.6 Etapa 6 — Documentación y comparación de modelos'}
    for old,new in replacements.items():
        p=find(old); p.text=new
        if old.startswith('1.3'): p.style=doc.styles['Heading 3']
    p=find('Limitación: a diferencia de Mendeley'); p.text=('Limitación: no hay un conjunto de prueba independiente del fold utilizado para seleccionar la mejor época. El mismo fold retenido permite elegir la época y reportar sus métricas; por ello el resultado puede ser optimista y debe interpretarse como comparación bajo este protocolo.')
    find('Flujo de datos entre las cuatro etapas').text = 'La figura inicial muestra las seis etapas del sistema, desde la preparación de los datos hasta la documentación, y su relación con el almacenamiento de resultados.'
    old_image=ps[27]; old_image._element.getparent().remove(old_image._element)
    env = find('Especificaciones técnicas del entorno de ejecución:')
    env.text = 'Especificaciones técnicas del entorno de Generalización:'
    source_doc = Document(ROOT.parent / 'Mendeley_ViT_DenseNet/6_documentacion/Documentacion_Tecnica_10_Modelos.docx')
    source_table = next(t for t in source_doc.tables if any('Sistema operativo' in c.text for r in t.rows for c in r.cells) and any('Acelerador' in c.text for r in t.rows for c in r.cells))
    heading = insert_before(env, 'Especificaciones técnicas de Mendeley (referencia)', 'Heading 3')
    heading._p.addnext(deepcopy(source_table._tbl))
    insert_before(find('Cada dominio se divide en 10 folds'), '1.2.1 Preparación de folds', 'Heading 3')
    ref=find('1.2.2 Configuración de los modelos')
    insert_before(ref, 'La preparación de los folds parte de las imágenes aceptadas. En paralelo, se prepara el modelo a partir del checkpoint de Mendeley correspondiente; ambas entradas se reúnen para entrenar cada fold.')
    p=find('Por cada combinación (modelo, dominio, fold)')
    p.text=('Cada combinación de modelo, conjunto y fold sigue un ciclo: se entrena la cabeza durante una época con nueve folds y luego se evalúa con el fold retenido. Se calcula el F1 macro y se compara con el mejor valor anterior. Si mejora, se actualiza el mejor resultado y la paciencia vuelve a cero; si no mejora, la paciencia aumenta. Después se comprueba si debe continuar: si quedan épocas y la paciencia no se agotó, se entrena la siguiente; al llegar a 30 épocas o acumular 8 sin mejora, se detiene y conserva el mejor resultado.')
    ref=find('Corridas experimentales por modelo y dominio')
    insert_before(ref, 'El proceso se repite diez veces, usando un fold retenido diferente en cada ejecución. Se selecciona la mejor época dentro de cada fold; no se elige un único mejor fold para representar todo el experimento.')
    insert_before(ref, 'Una sola corrida registra, por fold, el F1 de cada época y, de la mejor época, las etiquetas, las predicciones y las probabilidades. Con ellas se calculan las matrices y las curvas.')
    p=find('Con las predicciones del mejor epoch de cada fold')
    p.text=('Las etiquetas reales y las predicciones de la mejor época permiten construir la matriz de confusión y calcular las métricas de cada fold. La probabilidad de la clase no fresco se usa para obtener la curva ROC y el AUC. Todas las cifras por fold y pooled salen de esta misma corrida.')
    ref=find('2. Glosario de fórmulas')
    insert_before(ref,'Documentación de resultados','Heading 3')
    insert_before(ref,'La documentación reúne los resultados CSV, las gráficas PNG y el método. Todas las cifras salen de una sola corrida de 10 folds (linear_probe_con_test.py --protocolo original --folds 10).')
    doc.save(DOC)

if __name__ == '__main__': actualizar()
