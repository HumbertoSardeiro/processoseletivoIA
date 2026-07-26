# Projeto 3 — Detecção de Máscaras Faciais (YOLO)

## 💻 O Desafio Técnico

Desenvolva um modelo de **detecção de objetos** capaz de identificar, em uma
imagem com rostos, se cada pessoa está **usando máscara corretamente**, **sem
máscara**, ou **usando a máscara de forma incorreta** — localizando cada rosto
com uma bounding box.

Diferente dos Projetos 1 e 2 (onde você constrói uma CNN do zero), aqui o
objetivo é **adaptar e otimizar um framework de detecção real para Edge AI** —
uma competência bastante prática no dia a dia de Visão Computacional Embarcada,
já que a imensa maioria das aplicações de detecção em produção parte de um
modelo pré-treinado, não de uma arquitetura construída do zero.

> ⚠️ **Exceção importante:** ao contrário dos Projetos 1 e 2, aqui o uso de
> **pesos pré-treinados é permitido e esperado** (fine-tuning). Isso é
> intencional — este projeto avalia uma competência diferente: adaptar,
> treinar e exportar um framework de detecção real para o seu dataset.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**fine-tuning → validação → exportação → otimização para edge**

## 🎯 Conjunto de Dados

Este projeto já vem com um dataset **pronto para uso**, na pasta [`dataset/`](dataset/):
o **Face Mask Detection Dataset** ([Kaggle, andrewmvd](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection),
licença **CC0 1.0** — domínio público), já convertido do formato original (Pascal VOC)
para o formato esperado pelo Ultralytics YOLO.

- **853 imagens** de rostos, com bounding boxes anotadas
- **3 classes:** `with_mask`, `without_mask`, `mask_weared_incorrect`
- Já dividido em treino (~80%) e validação (~20%)
- ⚠️ O dataset é **desbalanceado** — a classe `mask_weared_incorrect` tem
  significativamente menos exemplos que as outras duas. Isso é uma
  característica real de datasets de detecção e não é um bug — comente esse
  ponto no seu relatório se perceber o modelo com dificuldade nessa classe.

Você **não precisa** baixar nada do Kaggle nem escrever código de conversão de
anotações — isso já está pronto em `dataset/`. Seu trabalho começa direto no
fine-tuning do modelo.

## ✅ Requisitos Obrigatórios

### Etapa 1 — Fine-tuning do Modelo (`train_model.py`)

Implemente, usando a biblioteca **Ultralytics** (YOLO):

- Carregamento do modelo pré-treinado **YOLO11n** (`YOLO("yolo11n.pt")`) —
  esta é a única exceção à regra de "sem modelos pré-treinados" do processo
  seletivo, válida especificamente para este projeto
- Fine-tuning no dataset fornecido (`dataset/data.yaml`), em **CPU**, com um
  número de épocas modesto (ex: 15-30 — YOLO converge relativamente rápido
  em fine-tuning, mesmo em CPU)
- Ao final do treino, copie os pesos resultantes (`runs/detect/train/weights/best.pt`)
  para a raiz desta pasta, com o nome **`model.pt`**

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.pt` treinado
- Exportação para **TensorFlow Lite** via `model.export(format="tflite")`
  (a Ultralytics gera automaticamente um arquivo `model.tflite` na mesma pasta)

> 💡 Na primeira execução, a Ultralytics pode instalar automaticamente
> dependências extras necessárias para a exportação (isso é esperado e pode
> levar alguns minutos).

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.pt`) usando `YOLO("model.tflite", task="detect")`
- Execução de inferência em pelo menos **5 imagens** de `dataset/images/val/`,
  **uma de cada vez** — o `model.tflite` exportado aceita apenas 1 imagem por
  chamada (batch=1), que é aliás o cenário real de uso em edge
- Exibição no terminal, para cada imagem, do número de detecções encontradas

> 💡 O Ultralytics salva automaticamente as imagens anotadas com as caixas
> preditas em `runs/detect/...` (pasta já ignorada pelo `.gitignore` — não
> precisa, nem deve, ser commitada). Abra essas imagens localmente pra conferir
> visualmente as predições antes de escrever o relatório.
>
> 💡 Essa etapa existe porque uma métrica agregada (mAP) pode esconder
> problemas que só aparecem olhando exemplos individuais — especialmente dado
> o desbalanceamento de classes deste dataset.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos nem a estrutura de `dataset/`.

```
projetos/3-deteccao-mascaras/
├── train_model.py         # ✏️ Fine-tuning do modelo
├── optimize_model.py      # ✏️ Exportação e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.pt               # 🤖 Gerado por você — deve ser commitado
├── model.tflite            # ⚡ Gerado por você — deve ser commitado
├── README.md               # 📝 Este arquivo (também usado como relatório)
└── dataset/                # 📦 Dataset já pronto (não modificar)
    ├── data.yaml
    ├── images/{train,val}/
    └── labels/{train,val}/
```

## ⚠️ Restrições e Considerações de Engenharia

- Modelo base: **YOLO11n** (variante *nano*, indicada para CPU/edge) — não use
  variantes maiores (s/m/l/x)
- Treinamento apenas em CPU
- Fine-tuning é permitido e esperado (única exceção às regras gerais do processo seletivo)
- **Não é esperada detecção perfeita**, especialmente na classe minoritária
  (`mask_weared_incorrect`) — o objetivo é demonstrar que o pipeline completo
  (fine-tuning → validação → exportação) funciona corretamente
- O tempo de treinamento e exportação deste projeto tende a ser **maior** que
  o dos Projetos 1 e 2 — reserve tempo extra para rodar localmente antes de enviar

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração de `model.pt` e `model.tflite`
- **Qualidade do modelo** — mAP50 no conjunto de validação acima do mínimo esperado
- **Edge AI** — exportação correta para `.tflite`
- **Documentação** — preenchimento adequado do relatório abaixo

---

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

Exportação para o formato de borda (executada em ambiente Linux — ver seção 5):

- **litert-torch 0.9.1**
- **ai-edge-litert 2.1.5**

### 3️⃣ Técnica de Otimização do Modelo

O modelo treinado (`model.pt`) foi exportado para o formato **LiteRT** — a nova
geração (e novo nome) do TensorFlow Lite — gerando o arquivo `model.tflite`. A
exportação foi feita com `model.export(format="tflite")`, que na versão atual da
Ultralytics é automaticamente redirecionada para o formato LiteRT. O `.tflite`
resultante é um artefato otimizado para execução *on-device* (mobile, embarcado,
edge), com um runtime enxuto adequado a hardware com restrição de recursos.

Além da exportação padrão (float32), também foi avaliada a **quantização
estática INT8** como técnica de compressão adicional — os resultados e as
limitações dessa avaliação estão descritos na seção 5.

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

Tamanho dos arquivos gerados (artefato entregue — exportação float32):

- **model.pt:** ~5,2 MB (5.455.834 bytes)
- **model.tflite:** ~10,1 MB (exportação LiteRT em float32)

Ambos os artefatos ficaram acima dos mínimos de aprovação (mAP50 ≥ 0,30 para o
`model.pt` e ≥ 0,20 para o `model.tflite`). A comparação de tamanho com a
alternativa quantizada (INT8) é discutida na seção 5.

### 5️⃣ Comentários Adicionais (Opcional)

**Avaliação da quantização INT8 (experimento).** Além da exportação padrão
(float32), foi testada a **quantização estática INT8** via
`model.export(format="tflite", int8=True, data="dataset/data.yaml")`, usando as
imagens de validação como conjunto de calibração. O ganho de tamanho foi
expressivo: o modelo INT8 ficou com **~2,9 MB**, contra ~5,2 MB do `model.pt` e
~10,1 MB do `.tflite` float32 — uma redução de **~3,5x** em relação ao float32,
tornando-o **menor que o próprio modelo original**. A precisão caiu de forma
controlada: mAP50 de 0,638 no modelo INT8, ainda muito acima do mínimo exigido.

**Por que o artefato entregue é o float32.** Apesar do INT8 ser mais leve, optei
por entregar a versão float32 por robustez de ambiente. A exportação INT8 exige
a instalação automática do pacote `litert-torch` no momento da conversão, e essa
instalação substitui a versão do PyTorch (de 2.13.0 para 2.12.1), gerando um
conflito de dependências (`torchvision` exige `torch==2.13.0`) que quebra a
conversão no ambiente de CI com o erro
`ImportError: cannot import name 'get_cuda_generator_meta_val'`. A exportação
float32 não depende dessa auto-instalação e é reproduzível de forma estável.
Trata-se de um trade-off consciente entre **tamanho** (favorável ao INT8) e
**reprodutibilidade** (favorável ao float32).

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

Saída do `run_inference.py` carregando o `model.tflite` (artefato de borda) e
rodando em 5 imagens do conjunto de validação, uma de cada vez:

```
Imagem                               Detecções  Detalhes
----------------------------------------------------------------------
maksssksksss105.jpg                         10  [10x with_mask]
maksssksksss107.jpg                          1  [1x with_mask]
maksssksksss11.jpg                          25  [23x with_mask, 1x mask_weared_incorrect, 1x without_mask]
maksssksksss113.jpg                          4  [3x with_mask, 1x without_mask]
maksssksksss12.jpg                          15  [13x with_mask, 2x without_mask]
----------------------------------------------------------------------
TOTAL                                       55
```

Comentário: ao abrir as imagens anotadas, as caixas apareceram bem posicionadas
sobre os rostos, inclusive na foto de multidão (`maksssksksss11.jpg`), com 25
detecções simultâneas. A grande maioria das detecções foi da classe `with_mask`,
coerente com o desempenho por classe na validação (mAP50 0,966 para `with_mask`).
A classe `mask_weared_incorrect` aparece raramente (1 detecção entre 55),
refletindo o forte desbalanceamento do dataset e a maior dificuldade do modelo
com essa categoria.

---

## 📄 Créditos do Dataset

Face Mask Detection Dataset — [Kaggle: andrewmvd/face-mask-detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection), licença CC0 1.0 (domínio público).