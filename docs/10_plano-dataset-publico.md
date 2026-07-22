# Plano — validar o pipeline com dataset público

> Objetivo desta fase: **provar que o pipeline de treino funciona** (preparação →
> patches → ramo de Fourier → convergência → avaliação → export ONNX → inferência)
> usando um dataset público pronto, **antes** de investir na coleta própria
> (`09_plano-de-captura-dataset.md`). É uma fase de de-risco, não o modelo final.
> Síntese de pesquisa; fontes no fim.

---

## 1. Objetivo e critério de sucesso

Não estamos buscando o melhor modelo aqui. Estamos validando que:

1. O `data_preparation` gera os patches no formato do repo sem erro.
2. O loop de treino roda, a loss cai, o modelo **converge** (overfita um subconjunto
   pequeno de propósito, como smoke-test).
3. A avaliação **subject-disjoint** produz APCER/BPCER coerentes.
4. O **sinal do rótulo está correto** (real→real) — ver seção 5, o erro nº1.
5. O modelo treinado exporta para ONNX e roda no nosso `predict.py`/`webcam.py`.

Se esses 5 passam, o pipeline está pronto para receber o dataset próprio.

---

## 2. Dataset escolhido: CelebA-Spoof (+ NUAA como smoke-test)

Ranking dos datasets viáveis para começar **hoje** (facilidade de obter + RGB +
baseado em imagem + licença tolerável):

| # | Dataset | Formato | Obtenção | Por quê |
|---|---|---|---|---|
| **1** | **CelebA-Spoof** | **Imagens RGB** | Google Drive oficial ou espelhos Kaggle (já com faces cortadas) | 625k imgs, 10k sujeitos, **bbox inclusa**, é o dataset de referência dos pipelines MiniFASNet |
| 2 | LCC-FASD | Imagens RGB | Kaggle, 1 clique | Pequeno, já usado em pipeline leve single-frame |
| 3 | NUAA Imposter | Imagens (grayscale) | Kaggle | Minúsculo → smoke-test em segundos. **Grayscale + só print** → não serve p/ avaliar qualidade |
| — | Replay-Attack, CASIA-FASD, OULU-NPU, SiW, MSU-MFSD | **Vídeo + EULA** | Burocrático | Ficam fora desta fase (atrito de EULA/e-mail institucional + extrair frames) |

**Decisão:** CelebA-Spoof como base principal (imagem, RGB, download direto, é o
dataset canônico do ecossistema MiniFASNet). NUAA como smoke-test opcional para
confirmar que o loop roda em segundos.

- **Licença CelebA-Spoof:** uso **não comercial / pesquisa**. OK para validar
  pipeline internamente; **não** pode virar produto. O modelo final de produção sai
  do nosso dataset próprio.
- **Atalho prático:** usar um espelho Kaggle **já com faces cortadas** para subir
  rápido (ex.: `attentionlayer241/celeba-spoof-for-face-antispoofing`).

---

## 3. Implementação de referência: fork `hairymax/Face-AntiSpoofing`

Não partir do zero. O fork **`hairymax/Face-AntiSpoofing`** é a implementação
pública mais completa que treina MiniFASNet a partir do CelebA-Spoof, com um
`data_preparation.py` pronto. Estratégia: **clonar, entender, adaptar** o que for
necessário para o nosso projeto.

Atenção a uma diferença de arquitetura: o fork **simplificou** o pipeline multi-escala
do minivision — gera **um único crop** parametrizado por `--bbox_inc` (fator de
expansão da bbox) em vez das 4 pastas (`org_1_80x60`, `1_80x80`, `2.7_80x80`,
`4_80x80`). Isso leva à decisão de design da seção 4.

---

## 4. Decisões de design (com recomendação)

### 4a. Multi-escala (A) vs single-crop (B)

| | (A) Fiel ao minivision | (B) Simplificado (hairymax) |
|---|---|---|
| Patches | 4 pastas de escala, 1 modelo por escala, ensemble | 1 escala (`bbox_inc`≈2.7), 1 modelo |
| Compatível com pesos originais | Sim | Parcial |
| Complexidade | Maior | Menor |

**Recomendação para esta fase: (B) single-crop.** O objetivo é validar o pipeline,
não reproduzir o ensemble. Um modelo, uma escala (~2.7), menos partes móveis. Quando
formos ao dataset próprio, avaliamos se vale o ensemble multi-escala.

### 4b. Binário (2 classes) vs 3 classes

- Os pesos pré-treinados do minivision usam **3 classes** (2D-fake / real / 3D-replay).
- Treinar com `num_classes=2` é mais simples (live/spoof direto), mas **os checkpoints
  originais não carregam** (a FC final muda de shape).

**Recomendação para esta fase: binário (`num_classes=2`), treinando do zero.** É o
que o hairymax faz e prova funcionar; é o caminho mais limpo para validar o loop.
Fine-tuning a partir dos pesos originais (com `strict=False`, ver `08`) fica para a
fase do dataset próprio, onde o domain shift justifica reusar o backbone.

### 4c. Do zero vs fine-tuning

Coerente com 4b: **do zero** nesta fase (CelebA-Spoof é grande o bastante, 625k imgs).
Fine-tuning é a estratégia da fase própria (dataset pequeno).

---

## 5. Mapeamento de rótulos — o erro nº1

> **CelebA-Spoof usa `0 = Live (real)`. O MiniFASNet, na inferência, espera
> `label == 1 = real`. Os sinais são OPOSTOS.** Inverter errado gera um modelo com
> acurácia alta no treino mas **predições invertidas** em produção — o pior tipo de
> bug, porque parece que funcionou.

Convenção original de cada dataset:

| Dataset | Real | Ataque |
|---|---|---|
| **CelebA-Spoof** | `spoof_type=0` (Live) | 1–3 print · 4–6 paper-cut · 7–9 replay · 10 mask |
| CASIA-FASD | seq 1,2,HR_1 | 3–8, HR_2..4 |
| Replay-Attack | pasta `real/` (freq. =1) | pasta `attack/` |
| NUAA | `ClientRaw/` | `ImposterRaw/` (print) |

**Mapa para as nossas pastas (binário, `1 = real`):**

```
pasta 0  ← todos os ataques   (CelebA spoof_type 1–10; NUAA imposter; …)
pasta 1  ← real / live        (CelebA spoof_type 0; NUAA client; …)
```

**Salvaguarda obrigatória:** depois de treinar, rodar o modelo em **imagens reais
conhecidas** (nossas próprias `images/custom/Elmer*.jpg`) e confirmar que dá "real".
Se der "fake" sistematicamente, o sinal foi invertido na preparação.

---

## 6. Armadilhas técnicas (confirmadas na pesquisa)

1. **Bbox do CelebA-Spoof está em escala 224.** Os `*_BB.txt` guardam `x y w h`
   relativos a uma imagem 224×224. **Obrigatório reescalar** para o tamanho real:
   `w_real = w * (largura_real/224)` (idem h/x/y). Esquecer → crops totalmente errados.
   Formato do `_BB.txt`: 4 números + um score no fim que se descarta.
2. **BGR em todo o pipeline.** O loader do minivision usa `cv2.imread` → BGR, e o
   `data_preparation.py` do hairymax mantém BGR (o `cvtColor(BGR2RGB)` fica comentado).
   Se preparar em RGB (PIL) mas inferir em BGR (OpenCV), os canais trocam → queda
   silenciosa. **Consistência: BGR em tudo** (é o default do nosso `predictor` também).
3. **Pixels em [0,255], sem normalizar.** Não inserir `Normalize(mean,std)` estilo
   ImageNet — quebra o modelo. Ver `01_pixel-range-0-255.md`.
4. **Clamp da bbox expandida.** Após expandir por 2.7×, o crop pode sair da imagem →
   fazer clamp/pad; descartar imagens onde o detector falha (não gerar crop preto).
5. **Ramo de Fourier é gerado online** por `DatasetFolderFT` (`dataset_folder.py`):
   grayscale → `fft2` → `fftshift` → `log(abs+1)` → min-max → resize 10×10 → alvo do
   MSELoss. **Reutilizar direto**, não precisa pré-computar.

---

## 7. Pipeline de preparação e treino (passos)

```
CelebA-Spoof (espelho Kaggle, faces já cortadas ou com _BB.txt)
    ↓  data_preparation (adaptar do hairymax)
    ├── ler _BB.txt, reescalar bbox de 224 → tamanho real   (armadilha #1)
    ├── mapear rótulo: spoof_type 0 → real; 1–10 → ataque    (seção 5)
    ├── crop bbox expandida ~2.7×, clamp, resize 80×80, BGR   (armadilhas #2,#4)
    ├── split SUBJECT-DISJOINT (por identidade CelebA)        (seção 8)
    └── salvar em datasets/RGB_Images/2.7_80x80/{0,1}/
    ↓  train (adaptar train_main.py p/ Python 3.10+, num_classes=2)
    ├── DatasetFolderFT (Fourier online)                      (armadilha #5)
    ├── loss = 0.5*CE + 0.5*MSE_ft
    └── checkpoint .pth
    ↓  convert_to_onnx.py (já pronto)
    ↓  predict.py / webcam.py  → validar sinal do rótulo      (seção 5)
```

Mudanças de código herdadas de `08_plano-de-treinamento.md` (portar `tensorboardX`,
remover `DataParallel` se GPU única, `num_classes` na config, carregamento de pesos).

---

## 8. Protocolo de validação

1. **Smoke-test** — subconjunto minúsculo (ex.: NUAA ou ~200 imgs CelebA). Confirmar
   que o loop roda e **overfita** (loss → ~0). Se não overfita num conjunto minúsculo,
   há bug no pipeline, não no dado.
2. **Split subject-disjoint** — CelebA-Spoof tem identidade por sujeito; dividir por
   sujeito (não por imagem). Nunca a mesma pessoa em train e test.
3. **Run de validação** — subconjunto maior do CelebA-Spoof, treino do zero binário.
4. **Métricas** — APCER (pior caso entre tipos de ataque), BPCER, ACER; threshold
   escolhido na validação (EER), não no teste. Ver `09` seção 7.
5. **Sanity-check do sinal** — rodar em `images/custom/Elmer*.jpg` reais → deve dar
   "real" (seção 5).
6. **Export + inferência** — `convert_to_onnx.py` → `webcam.py` com o novo modelo.

Sucesso = os 5 critérios da seção 1 satisfeitos. A partir daí, o mesmo pipeline
recebe o dataset próprio, agora com **fine-tuning** (não do zero) e possivelmente
**3 classes / multi-escala** se justificar.

---

## 9. Próximos passos (código)

- [ ] 1. Baixar CelebA-Spoof (espelho Kaggle com faces cortadas) + NUAA (smoke-test)
- [ ] 2. Clonar `hairymax/Face-AntiSpoofing` como referência
- [ ] 3. `prepare_public_dataset.py` — adaptar `data_preparation.py`: rescale bbox 224,
        mapeamento de rótulo, crop 2.7× BGR, split subject-disjoint → `RGB_Images/`
- [ ] 4. `train.py` adaptado (Python 3.10+, `num_classes=2`, do zero)
- [ ] 5. Smoke-test → run de validação → métricas
- [ ] 6. Export ONNX + sanity-check do sinal em `webcam.py`
- [ ] 7. Só então: partir para coleta própria (`09`) reusando este pipeline

---

## Fontes

- Fork de referência (CelebA-Spoof → MiniFASNet): https://github.com/hairymax/Face-AntiSpoofing
- CelebA-Spoof (dataset + anotações, live=0): https://github.com/ZhangYuanhan-AI/CelebA-Spoof · https://arxiv.org/abs/2007.12342
- Espelho Kaggle com faces cortadas: https://www.kaggle.com/datasets/attentionlayer241/celeba-spoof-for-face-antispoofing
- LCC-FASD (Kaggle): https://www.kaggle.com/datasets/faber24/lcc-fasd · pipeline leve: https://github.com/kprokofi/light-weight-face-anti-spoofing
- NUAA (Kaggle): https://www.kaggle.com/datasets/aleksandrpikul222/nuaaaa
- MiniFASNet original + loader Fourier: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/src/data_io/dataset_folder.py
- Clean-weights / ONNX: https://github.com/facenox/face-antispoof-onnx
