"""Prueba de Wilcoxon pareada: ViT-Base/16 frente a DenseNet-121, por dominio.

Usa el F1 de cada uno de los 10 folds de la corrida de linear_probe_con_test.py
(3_resultados/{trucha,rohu}/linear_probe_10fold/resumen.csv); con 5 folds el
p-valor más bajo posible sería 0.0625 y nunca alcanzaría α = 0.05.

Uso: python 1_codigo/wilcoxon_vit_vs_densenet.py
Salida: 3_resultados/wilcoxon_vit_vs_densenet.csv
"""
import os

import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALFA = 0.05


def main():
    filas = []
    for dominio in ("trucha", "rohu"):
        r = pd.read_csv(os.path.join(ROOT, "3_resultados", dominio, "linear_probe_10fold", "resumen.csv"))
        r = r[r["fold"].astype(str).str.isdigit()].assign(fold=lambda d: d["fold"].astype(int))
        vit = r[r["model"] == "vit_base_patch16_224"].sort_values("fold")["test_f1"].to_numpy()
        dnc = r[r["model"] == "densenet121"].sort_values("fold")["test_f1"].to_numpy()
        res = wilcoxon(vit, dnc)  # bilateral, exacta para n pequeño
        filas.append({"dominio": dominio, "n_folds": len(vit), "folds_vit_mejor": int((vit > dnc).sum()),
                      "media_f1_vit": round(vit.mean(), 2), "media_f1_densenet": round(dnc.mean(), 2),
                      "W": res.statistic, "p_valor": round(res.pvalue, 4),
                      "significativo_alfa_0_05": res.pvalue < ALFA})
    out = pd.DataFrame(filas)
    out.to_csv(os.path.join(ROOT, "3_resultados", "wilcoxon_vit_vs_densenet.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
