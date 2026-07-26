import shutil
from pathlib import Path

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Projeto 3 - Deteccao de Mascaras Faciais (Fine-tuning do YOLO11n)
# ---------------------------------------------------------------------------


def main():
    # 1. Modelo base pre-treinado (a Ultralytics baixa yolo11n.pt automaticamente)
    model = YOLO("yolo11n.pt")

    # 2. Fine-tuning em CPU.
    results = model.train(
        data="dataset/data.yaml",
        epochs=20,
        imgsz=640,
        batch=16,
        device="cpu",
        patience=10,
        seed=42,
        verbose=True,
    )

    # 3. Copia os melhores pesos para model.pt (na raiz da pasta do projeto)
    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy(best, "model.pt")
    print(f"\nPesos treinados copiados para: model.pt")

    # 4. Validacao explicita sobre o conjunto de validacao (para o relatorio)
    trained = YOLO("model.pt")
    metrics = trained.val(data="dataset/data.yaml", split="val", verbose=True)
    print("\n" + "=" * 50)
    print(f"mAP50     (validacao): {metrics.box.map50:.4f}")
    print(f"mAP50-95  (validacao): {metrics.box.map:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()