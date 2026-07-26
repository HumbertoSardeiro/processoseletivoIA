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
    preferidos = [c for c in candidatos if "float32" in c.name]
    return (preferidos or candidatos)[0]


def main():
    model = YOLO("model.pt")
    exportado = model.export(format="tflite", imgsz=640)
    origem = _achar_tflite(exportado)

    destino = Path("model.tflite")
    if origem.resolve() != destino.resolve():
        shutil.copy(origem, destino)

    print(f"model.tflite gerado: {destino.resolve()} ({destino.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()