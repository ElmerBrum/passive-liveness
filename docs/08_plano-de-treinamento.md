# Plano de treinamento — MiniFASNet com dataset próprio

> Este plano foi verificado contra o código de treino real do repo original
> (`../Silent-Face-Anti-Spoofing/src/train_main.py`, `MultiFTNet.py`,
> `data_io/dataset_folder.py`, `data_io/dataset_loader.py`). As afirmações
> abaixo refletem o que o código de fato faz, não só o README.

---

## 1. Por que treinar com a câmera própria

O modelo original foi treinado pela Minivision em câmeras Android específicas.
Cada câmera tem características que afetam a **textura** capturada pela rede — e
é exatamente a textura que separa "pele real" de "pele em papel/tela":

| Fator | Impacto |
|---|---|
| Resposta de cor (ISP) | Saturação e balanço de branco diferentes |
| Compressão de vídeo | Artefatos JPEG/H264 específicos por fabricante |
| Foco e nitidez | Sharpening aplicado pelo driver |
| Ruído | Padrão de noise único do sensor |
| Distorção de lente | Barrel/pincushion distortion |

Mudar a câmera muda as texturas → a fronteira de decisão aprendida fica deslocada.
Isso é *domain shift*, e é o motivo nº1 para retreinar.

---

## 2. Como o modelo realmente treina (arquitetura de treino)

Isto é essencial para entender por que certas estratégias de fine-tuning
funcionam e outras não.

### Duas perdas, dois ramos independentes

O treino NÃO usa a `MiniFASNet` diretamente — usa a `MultiFTNet`
(`MultiFTNet.py:34`), que envolve a MiniFASNet e adiciona um segundo ramo:

```
entrada (patch 80x80)
    │
    ├──► MiniFASNet ──► cls (logits, num_classes)   ──► CrossEntropyLoss(cls, label)
    │                                                        (ramo de classificação)
    │
    └──► FTGenerator ──► mapa de features           ──► MSELoss(mapa, espectro_FFT)
                                                             (ramo auxiliar de Fourier)

loss_total = 0.5 * loss_cls + 0.5 * loss_ft      (train_main.py:112)
```

- **Ramo de classificação**: as classes `0/1/2` das pastas do dataset, via
  `CrossEntropyLoss` (`train_main.py:109`).
- **Ramo de Fourier**: o `FTGenerator` (`MultiFTNet.py:12`) tenta **reproduzir o
  espectro FFT** da imagem. O alvo é calculado online por `generate_FT()`
  (`dataset_folder.py:52`): FFT 2D da imagem em cinza, magnitude em log, normalizada.
  Comparado via `MSELoss` (`train_main.py:110`).

> **Correção de um erro comum:** a supervisão de Fourier **não** é "a classe 2".
> São mecanismos separados. A classe 2 é uma classe de classificação genuína; o
> Fourier é uma regressão de espectro num ramo à parte que só existe em modo treino.

### A classe 2

O repo original define 3 pastas de classe (`0/`, `1/`, `2/`) mas **não documenta a
semântica exata** de cada uma. O que sabemos com certeza pelo código:

- São 3 classes reais de classificação (`num_classes=3`).
- Na **inferência**, `label == 1` = rosto real; qualquer outra = fake
  (é o que nosso `predict` faz, restringindo o argmax a `[0,1]`).

Para o nosso dataset, a escolha mais simples e segura é usar **apenas 2 classes**
(`0`=fake, `1`=real) e treinar com `num_classes=2`, a menos que tenhamos um motivo
concreto para uma terceira categoria. Isso simplifica o rótulo e a avaliação.
(Ver seção 8 sobre a mudança de código necessária.)

### O `FTGenerator` não está nos pesos liberados

Os `.pth` liberados (`resources/models/*.pth`) contêm **apenas o submodelo
MiniFASNet de inferência** — não incluem o `FTGenerator`. Isso tem consequência
direta no fine-tuning (seção 5).

### O treino só constrói `MiniFASNetV2SE`

`MultiFTNet.py:39` instancia **sempre** `MiniFASNetV2SE`, independente do
`patch_info`. Ou seja, o `train.py` open-source **não reproduz** o ensemble liberado
(`MiniFASNetV2` + `MiniFASNetV1SE`) sem alterar `_define_network`
(`train_main.py:118`). Precisamos decidir a arquitetura explicitamente (seção 8).

---

## 3. Coleta de dados

### Protocolo — rosto real (bona fide)

- Gravar **com a câmera alvo** (a mesma de produção).
- **Máximo de sujeitos possível** — ver a nota crítica sobre split abaixo. 5 pessoas
  é o mínimo absoluto e **não** permite um protocolo de avaliação honesto; mire em
  **15–30+** se a meta for generalização real.
- Variações por pessoa:
  - Iluminação: ambiente, lateral, contra-luz suave
  - Ângulo: frontal, ±15°, ±30° (o modelo assume rosto < 30°)
  - Distância: 30cm, 60cm, 100cm
  - Expressões: neutro, sorrindo, falando
- 10–15 s por clip (~150–225 frames a 15fps).

### Protocolo — ataques (fake)

- Criar os ataques a partir das **mesmas capturas dos sujeitos reais** (mesma
  identidade em real e fake → o modelo aprende **textura**, não identidade).
- Tipos de ataque, por prioridade:

| Tipo | Coleta | Prioridade |
|---|---|---|
| Foto impressa (papel fosco) | Baixa | Alta |
| Foto impressa (papel brilhante) | Baixa | Alta |
| Replay em tela de celular | Baixa | Alta |
| Replay em monitor | Baixa | Média |
| Replay em tablet | Baixa | Média |
| Máscara 2D (papel recortado) | Média | Média |
| Máscara 3D (silicone) | Alta | Baixa |

- Variar distância e ângulo como no real.

### Balanceamento

- Manter ~1:1 real/fake no treino. Se houver mais fake, subsamplear.

---

## 4. Split do dataset — a parte que mais importa

> **Este é o erro nº1 em projetos de anti-spoofing.** Um split ingênuo produz
> métricas ótimas que desmoronam em produção.

Duas regras não-negociáveis:

1. **Subject-disjoint**: a mesma pessoa **nunca** aparece em mais de um split.
   Se o João está no treino, todos os frames do João (real e fake) ficam no treino.
   Caso contrário o modelo memoriza o rosto, não a vivacidade.

2. **Clip-disjoint**: frames do **mesmo vídeo** nunca se dividem entre splits.
   Frames consecutivos são quase idênticos — colocá-los em train e val vaza
   informação e infla a métrica.

Consequência prática: o split é feito **por sujeito**, não por frame.

```
Ex.: 20 sujeitos → 14 treino / 3 val / 3 teste  (70/15/15 por PESSOA)
```

Com poucos sujeitos (ex.: 5), não existe split honesto — o "teste" de 1 pessoa não
tem valor estatístico. Priorize coletar mais sujeitos antes de treinar.

### Estrutura de pastas (formato exigido pelo repo)

O treino carrega os patches via `ImageFolder`, então a estrutura de pastas
**é** o rótulo. Cada `patch_info` (escala) tem sua própria árvore:

```
datasets/RGB_Images/
├── org_1_80x60/        ← imagem inteira redimensionada (contexto global)
│   ├── 0/  (fake)
│   └── 1/  (real)
├── 1_80x80/            ← patch scale=1.0
│   ├── 0/
│   └── 1/
├── 2.7_80x80/          ← patch scale=2.7
│   ├── 0/
│   └── 1/
└── 4_80x80/            ← patch scale=4.0
    ├── 0/
    └── 1/
```

(Usando 2 classes. Se optar por 3, adicione a pasta `2/`.)

O nome da pasta de escala segue a convenção dos `.pth`:
`<scale>_<h>x<w>` → `2.7_80x80` → `2.7_80x80_MiniFASNetV2.pth`.

---

## 5. Estratégia de fine-tuning (revisada)

### Por que NÃO congelar o backbone

A intuição de "treinar só a cabeça" (`linear+bn+prob`) é a **menos eficaz** aqui:

- A cabeça tem pouquíssimos parâmetros e opera sobre um embedding já formado.
- O que muda entre câmeras é **textura e frequência** — features das camadas
  **convolucionais iniciais/médias**.
- Congelar o backbone = congelar exatamente onde o domain shift acontece.

### Abordagem recomendada, em ordem de esforço

| Nível | O que treinar | Quando usar |
|---|---|---|
| A | Todas as convs + cabeça, LR baixo (`1e-3`) a partir dos pesos liberados | **Padrão recomendado** |
| B | Só as stages finais (`conv_4`,`conv_45`,`conv_5`,`conv_6*`) + cabeça | Dataset muito pequeno, evitar overfit |
| C | Treino do zero (init aleatório) | Só se tiver milhares de amostras + GPU |

Comece pelo **nível A**: descongelar tudo, LR baixo, poucas épocas. É o que melhor
corrige textura sem destruir as features de face já aprendidas.

### Mecânica concreta (o que o código precisa fazer)

Como os `.pth` não têm o `FTGenerator`, o carregamento é parcial:

```python
# pseudo-código do _init_model_param adaptado
model = MultiFTNet(num_classes=2, conv6_kernel=kernel)   # FTGenerator init aleatório

pretrained = torch.load("resources/models/2.7_80x80_MiniFASNetV2.pth",
                        map_location=device, weights_only=False)
# remove prefixo "module." se existir (ver predictor.py)
# carrega SÓ no submodelo MiniFASNet, ignorando chaves ausentes/extras:
missing, unexpected = model.model.load_state_dict(pretrained, strict=False)
# 'missing' deve conter apenas chaves do FTGenerator (esperado)
```

Pontos de atenção:
- `strict=False` é necessário porque o `FTGenerator` não existe nos pesos.
- Se mudar `num_classes` (3→2), a camada `prob` muda de shape → ela **não** carrega
  e é reinicializada. Isso é o esperado ao trocar o número de classes.
- Escala de pixel **[0, 255]** deve ser mantida no treino, igual à inferência
  (ver `01_pixel-range-0-255.md`). O loader original já faz isso.

---

## 6. Configuração de treinamento

Baseado no `default_config.py` original, ajustado para fine-tuning:

```python
lr             = 1e-3       # << original 1e-1 — baixo para preservar os pesos
milestones     = [5, 10, 15]
gamma          = 0.1
epochs         = 20
momentum       = 0.9
weight_decay   = 5e-4       # igual ao original (train_main.py:38)
batch_size     = 64         # GPU típica; reduzir se faltar VRAM
num_classes    = 2          # fake / real (ver seção 2)
embedding_size = 128
```

### Augmentation — cuidado específico de anti-spoofing

O loader já aplica augmentation (`dataset_loader.py:14`):
`RandomResizedCrop(0.9–1.1)`, `ColorJitter(0.4/0.4/0.4/0.1)`, `RandomRotation(10)`,
`RandomHorizontalFlip`.

> **Atenção:** `ColorJitter` agressivo pode **prejudicar** anti-spoofing, porque
> cor e textura *são* o sinal que distingue real de fake. Considere **reduzir** o
> `ColorJitter` (ex.: 0.1) ao fine-tunar num domínio de câmera específico. Flip e
> rotação leve são seguros.

### Hardware

- **GPU fortemente recomendada.** O config original usa `batch_size=1024` e
  `num_workers=16` — é um pipeline pensado para GPU. Em CPU o treino é impraticável
  para qualquer dataset de tamanho realista. Estimativas de "1–2h em CPU" não são
  confiáveis.

---

## 7. Avaliação

Métrica padrão de liveness (ISO/IEC 30107-3):

```
APCER = fração de ATAQUES aceitos como real   (falso negativo de ataque)
BPCER = fração de REAIS rejeitados como fake   (falso positivo de ataque)
ACER  = (APCER + BPCER) / 2
```

- Avaliar **sempre no split de teste subject-disjoint** (seção 4).
- Reportar **APCER por tipo de ataque** (paper fosco vs brilhante vs replay…) —
  a média esconde o pior caso.
- Curva **ROC + AUC**.
- **Escolher o limiar** (o `--threshold` do nosso `predict.py`/`webcam.py`) **no
  split de validação**, mirando um APCER-alvo, e só então medir no teste. Não
  reusar o `0.7` arbitrário atual sem calibrar.

Referência do repo (modelo APK open-source): FPR=1e-5 → TPR=97.8%.
Meta inicial razoável com dataset próprio: **ACER < 5%** no teste.

---

## 8. Mudanças de código necessárias

Para adaptar o treino original ao nosso projeto (Python 3.10+ e nossas escolhas):

1. **`_define_network`** (`train_main.py:118`): permitir escolher a arquitetura
   (`MiniFASNetV2`, `MiniFASNetV1SE`, …) por `patch_info`, em vez do `V2SE` fixo.
   Necessário para reproduzir o ensemble liberado.
2. **Carregamento parcial dos pesos** com `strict=False` (seção 5).
3. **`num_classes`**: decidir 2 vs 3 e ajustar config + estrutura de pastas.
4. **Reduzir `ColorJitter`** no `dataset_loader` (seção 6).
5. **Portar dependências**: `tensorboardX` → `torch.utils.tensorboard`;
   remover `DataParallel` se treino em GPU única; `num_workers` conforme a máquina.
6. **Split subject-disjoint** no script de preparação (não é responsabilidade do
   `train.py` — precisa ser garantido na geração do dataset).

---

## 9. Próximos passos (ordem)

- [ ] 1. `collect_data.py` — captura de clips real/fake via webcam, organizados por sujeito
- [ ] 2. `prepare_dataset.py` — extrai frames, detecta face, gera patches multi-escala
       e faz o **split subject-disjoint**
- [ ] 3. Adaptar `train.py`/`train_main.py` (mudanças da seção 8)
- [ ] 4. Fine-tuning nível A dos modelos escolhidos
- [ ] 5. Avaliar ACER/APCER/BPCER no teste + calibrar limiar na validação
- [ ] 6. Exportar pesos para ONNX (`convert_to_onnx.py` já pronto)
- [ ] 7. Validar no `webcam.py` com os novos modelos

---

## Referências

- Código de treino original: `../Silent-Face-Anti-Spoofing/src/train_main.py`
- Arquitetura de treino: `../Silent-Face-Anti-Spoofing/src/model_lib/MultiFTNet.py`
- Loader + Fourier: `../Silent-Face-Anti-Spoofing/src/data_io/dataset_folder.py`
- Detector RetinaFace (insightface): https://github.com/deepinsight/insightface
- ISO/IEC 30107-3: Presentation Attack Detection — define APCER/BPCER/ACER
