"""Compara las cuatro configuraciones DBSCAN de la Actividad 2."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base import DBSCAN, SEED, StandardScaler, datos_segmentacion


def main():
    datos = datos_segmentacion()
    escalados = StandardScaler().fit_transform(datos)
    configuraciones = [
        ("Original", 0.28, 5),
        ("A", 0.20, 5),
        ("B", 0.45, 5),
        ("C", 0.28, 8),
    ]
    print(f"SEED={SEED} | observaciones={len(datos)} | columnas={list(datos)}")
    print("Experimento | eps | min_samples | grupos | ruido | tamaños de grupos")
    for nombre, eps, min_samples in configuraciones:
        etiquetas = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(escalados)
        grupos = sorted(set(etiquetas) - {-1})
        tamanos = [int((etiquetas == grupo).sum()) for grupo in grupos]
        print(f"{nombre} | {eps:.2f} | {min_samples} | {len(grupos)} | {(etiquetas == -1).sum()} | {tamanos}")


if __name__ == "__main__":
    main()
