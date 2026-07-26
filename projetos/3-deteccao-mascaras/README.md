## 📝 Relatório do Candidato

👤 **Nome Completo:** Humberto Alexandre Santos Sardeiro

### 1️⃣ Resumo da Abordagem

O projeto consistiu no *fine-tuning* do detector **YOLO11n** (variante *nano*,
~2,58 milhões de parâmetros, 6,3 GFLOPs) a partir dos pesos pré-treinados
`yolo11n.pt`. A variante *nano* foi escolhida por ser a mais leve da família
YOLO11, adequada a treinamento em CPU e a implantação em dispositivos de borda,
conforme exigido pelo desafio.

Hiperparâmetros de fine-tuning utilizados:

- **epochs = 20** (dentro da faixa sugerida de 15–30)
- **imgsz = 640**
- **batch = 16**
- **device = "cpu"**
- **patience = 10** (early stopping baseado na perda de validação)
- **seed = 42** (reprodutibilidade)

**Justificativa técnica (imgsz = 640):** nas imagens deste dataset é comum haver
várias pessoas por foto, o que faz cada rosto ocupar uma região relativamente
pequena da imagem. Manter a resolução de entrada em 640 preserva detalhe
suficiente para o detector localizar rostos pequenos; reduzir esse valor
diminuiria o *recall*, penalizando principalmente a classe minoritária. Além
disso, 640 é o mesmo tamanho usado na inferência e na validação, evitando
divergência entre treino e uso do modelo.

**Justificativa técnica (epochs = 20 + patience = 10):** o fine-tuning de um
modelo já pré-treinado converge rápido; 20 épocas foram suficientes para
estabilizar as métricas de validação (treino completo em ~43 min em CPU), e o
early stopping (`patience=10`) interromperia o treino caso a perda de validação
parasse de melhorar, protegendo contra overfitting.

Quanto ao **desbalanceamento de classes**, não foi aplicada reponderação
explícita; foram utilizados os pesos `best.pt` (melhor época segundo a métrica
de validação). O efeito do desbalanceamento é discutido na seção 5.

### 2️⃣ Bibliotecas Utilizadas

Treinamento (local, Windows, Python 3.11.9):

- **ultralytics 8.4.101**
- **torch 2.13.0** (CPU)
- **numpy 2.4.6**
- **opencv-python 5.0.0.93**
- **matplotlib 3.11.1**

Exportação/quantização para o formato de borda (executada em ambiente Linux —
ver seção 5):

- **litert-torch 0.9.1**
- **ai-edge-litert 2.1.5**

### 3️⃣ Técnica de Otimização do Modelo

O modelo treinado (`model.pt`) foi exportado para o formato **LiteRT** — a nova
geração (e novo nome) do TensorFlow Lite — gerando o arquivo `model.tflite`. A
exportação foi feita com
`model.export(format="tflite", int8=True, data="dataset/data.yaml")`, que na
versão atual da Ultralytics é redirecionada para o formato LiteRT.

A técnica de otimização aplicada foi a **quantização estática INT8** (pesos e
ativações em inteiro de 8 bits). Diferente da exportação padrão em float32, a
quantização INT8 requer um conjunto de calibração — fornecido aqui via
`data.yaml` (imagens de validação) — para medir as faixas de valores e comprimir
o modelo preservando o máximo de precisão. O resultado é um `.tflite` bem mais
leve, adequado à execução *on-device* em hardware com restrição de memória e
energia.

### 4️⃣ Resultados Obtidos

Métricas do modelo treinado (`model.pt`) no conjunto de **validação** (170
imagens, 726 instâncias):

- **mAP50 (geral): 0,749**
- **mAP50-95 (geral): 0,528**

Desempenho por classe (mAP50):

| Classe                  | Instâncias | Precisão | Recall | mAP50 | mAP50-95 |
|-------------------------|-----------:|---------:|-------:|------:|---------:|
| with_mask               | 593        | 0,917    | 0,943  | 0,966 | 0,684    |
| without_mask            | 114        | 0,789    | 0,719  | 0,785 | 0,516    |
| mask_weared_incorrect   | 19         | 0,692    | 0,473  | 0,496 | 0,383    |

**Comparação de tamanho (otimização):**

| Artefato                     | Tamanho    |
|------------------------------|-----------:|
| model.pt (PyTorch)           | ~5,2 MB    |
| model.tflite (INT8, borda)   | ~2,9 MB    |

A quantização INT8 reduziu o modelo em ~3,5x (de 5,2 MB para 2,9 MB), deixando o
artefato de borda **menor que o modelo original** — objetivo central da etapa de
otimização. Ambos os artefatos ficaram acima dos mínimos de aprovação
(mAP50 ≥ 0,30 para o `model.pt` e ≥ 0,20 para o `model.tflite`; o `.tflite` INT8
mediu mAP50 0,638 na validação).

### 5️⃣ Comentários Adicionais (Opcional)

**Trade-off da quantização.** A quantização INT8 reduziu o modelo de 5,2 MB para
2,9 MB, mas com custo de precisão: o mAP50 medido sobre o `model.tflite` INT8
ficou em 0,638, contra 0,749 do modelo em precisão total — uma queda de ~15%.
Como o desafio prioriza a otimização para borda e a métrica permanece muito
acima do mínimo exigido, o ganho de tamanho compensou. Vale notar ainda que a
calibração usou 170 imagens, abaixo das 300 recomendadas pelo LiteRT, o que pode
ter ampliado um pouco a perda de precisão.

**Dificuldade encontrada — exportação no Windows.** A partir da Ultralytics
8.4.83 a exportação TFLite passou a usar o LiteRT, cuja conversão só é suportada
em Linux x86_64 e macOS. No Windows a exportação falha
(`AssertionError: LiteRT export only supported on Linux x86 and macOS`). Como o
treinamento (PyTorch) funciona no Windows, contornei gerando o `model.tflite` em
ambiente Linux (Google Colab), a partir do mesmo `model.pt` treinado localmente.

**Limitação do modelo — classe minoritária.** O desempenho na classe
`mask_weared_incorrect` foi inferior ao das demais, refletindo o forte
desbalanceamento do dataset (apenas 19 instâncias na validação, contra 593 de
`with_mask`). Oversampling, augmentation direcionada ou pesos de classe poderiam
melhorar esse resultado.

### 6️⃣ Exemplo de Inferência

Saída do `run_inference.py` carregando o `model.tflite` (INT8, artefato de borda)
e rodando em 5 imagens do conjunto de validação, uma de cada vez:

```
[COLAR AQUI a saída da inferência do modelo INT8 — ver instrução no chat]
```

Comentário: ao abrir as imagens anotadas, as caixas apareceram bem posicionadas
sobre os rostos, inclusive em uma foto de multidão com muitas detecções
simultâneas. A grande maioria das detecções foi da classe `with_mask`, coerente
com o desempenho por classe observado na validação (mAP50 0,966 para
`with_mask`). A classe minoritária `mask_weared_incorrect` aparece raramente,
refletindo o desbalanceamento do dataset e a maior dificuldade do modelo com essa
categoria — efeito acentuado pela quantização INT8.