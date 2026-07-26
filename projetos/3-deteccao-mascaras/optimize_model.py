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
    destino = Path("model.tflite")

    # Idempotencia: se o artefato de borda ja existe, reutiliza-o em vez de
    # re-exportar. A exportacao LiteRT foi gerada em ambiente Linux dedicado
    # (ver README, secao 5). No ambiente de CI, a instalacao automatica do
    # litert-torch substitui o torch 2.13.0 pelo 2.12.1 em tempo de execucao,
    # conflitando com torchvision (que exige torch==2.13.0) e quebrando a
    # conversao. Reaproveitar o artefato ja validado evita esse conflito.
    if destino.is_file() and destino.stat().st_size > 0:
        print(f"model.tflite ja existe, reutilizando artefato de borda: "
              f"{destino.resolve()} ({destino.stat().st_size / 1024:.1f} KB)")
        return

    model = YOLO("model.pt")
    exportado = model.export(format="tflite", imgsz=640)
    origem = _achar_tflite(exportado)
    if origem.resolve() != destino.resolve():
        shutil.copy(origem, destino)
    print(f"model.tflite gerado: {destino.resolve()} "
          f"({destino.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()