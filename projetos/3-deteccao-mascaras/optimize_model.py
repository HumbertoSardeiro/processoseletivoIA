import shutil
from pathlib import Path

from ultralytics import YOLO


def _achar_tflite(dica):
    dica = Path(dica)
    if dica.is_file() and dica.suffix == ".tflite":
        return dica
    raiz = dica if dica.is_dir() else Path(".")
    candidatos = sorted(raiz.rglob("*.tflite")) or sorted(Path(".").rglob("*.tflite"))
    if not candidatos:
        raise FileNotFoundError("Nenhum .tflite foi gerado pela exportacao.")
    # prefere o arquivo quantizado (int8), se houver
    int8 = [c for c in candidatos if "int8" in c.name]
    return (int8 or candidatos)[0]


def main():
    model = YOLO("model.pt")
    # Quantizacao INT8: reduz o tamanho do modelo (~3.5x menor).
    # Usa o data.yaml para coletar imagens de calibracao.
    exportado = model.export(format="tflite", imgsz=640, int8=True, data="dataset/data.yaml")
    origem = _achar_tflite(exportado)

    destino = Path("model.tflite")
    if origem.resolve() != destino.resolve():
        shutil.copy(origem, destino)

    print(f"model.tflite (INT8) gerado: {destino.resolve()} ({destino.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()