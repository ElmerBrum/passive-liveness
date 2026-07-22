# Plano de captura do dataset

> Síntese de pesquisa sobre protocolos consolidados (OULU-NPU, Replay-Attack,
> CASIA-FASD, SiW, CelebA-Spoof) e ISO/IEC 30107-3, adaptada ao **nosso caso**:
> 1 webcam 1280×720, poucos sujeitos, modelo passivo single-frame (MiniFASNet).
> Complementa `08_plano-de-treinamento.md`. Fontes no fim do arquivo.

---

## 0. Escopo e limitação assumida (leia primeiro)

Usaremos **uma única câmera**. Isso tem uma consequência que precisa ser aceita
conscientemente:

> O modelo ficará **casado a este sensor**. As métricas que medirmos **não**
> predizem desempenho em outra webcam. Trocar a câmera em produção invalida a
> avaliação. Isso é aceitável para "produção específica" — mas fica registrado
> como limitação de domínio, não como bug a resolver depois.

Consequência prática: não conseguimos fazer avaliação *cross-device* (Protocolo 3
do OULU-NPU é impossível). Focamos em generalização de **iluminação, pose, PAI e
sujeito** — que são os eixos que conseguimos variar.

---

## 1. Princípio central: variar TUDO igualmente entre real e ataque

O erro nº1 em PAD caseiro não é falta de dados — é **correlação acidental** entre a
classe e algum fator de captura. Se todos os reais têm boa luz e todos os ataques
foram gravados noutra sessão, o modelo aprende **luz/fundo**, não vivacidade. Quando
o atalho some no teste, ele colapsa.

**Regra única que rege todo o resto:** cada condição (iluminação, fundo, distância,
pose, formato de arquivo) deve aparecer **tanto em real quanto em ataque**, em
proporções parecidas. A variação é a mesma; só muda o que está na frente da câmera.

---

## 2. A matriz de variação

Capturamos o produto cartesiano (aproximado) destes eixos, **idêntico** para real e
para cada PAI:

| Eixo | Valores | Origem da recomendação |
|---|---|---|
| **Iluminação** | frontal difusa · lateral 45° · contraluz · penumbra (pouca luz) | OULU 3 sessões, CelebA-Spoof 4 condições |
| **Pose (yaw)** | ±30° denso, cauda até ±45° | SiW; sistema real olha ~frontal |
| **Pose (pitch/roll)** | pitch ±25°, roll ±15° | CelebA-Spoof |
| **Distância** | ~40cm · ~70cm · ~120cm | SiW (sessão de distância variável) |
| **Fundo** | 3–4 fundos distintos | OULU/Replay |
| **Expressão** | neutro · falando · sorrindo | SiW (evita overfit a face estática) |

Não precisamos capturar todas as combinações explosivamente — dentro de um clip de
10–15s o sujeito **varia pose/distância/expressão naturalmente**. O que fixamos por
clip é **iluminação + fundo + PAI**; o resto varia dentro do clip.

---

## 3. Instrumentos de ataque (PAIs) que vamos coletar

Prioridade alta primeiro. Cada PAI é capturado sob a mesma matriz da seção 2.

| PAI | Especificação | Prioridade |
|---|---|---|
| `print-matte` | Foto do sujeito, papel **fosco**, tamanho ~real da face | Alta |
| `print-glossy` | Mesma foto, papel **brilhante/fotográfico** | Alta |
| `replay-phone` | Foto/vídeo exibido em **celular** | Alta |
| `replay-monitor` | Exibido em **monitor/tablet** | Média |

Boas práticas de fabricação (evitar vieses acidentais):

- **Print**: use ≥2 acabamentos de papel (fosco reflete luz de forma diferente do
  brilhante). Capture **com e sem reflexo especular** — não force sempre reflexo
  (senão "brilho = fake") nem sempre evite. Inclua papel **plano e levemente
  curvado** (CASIA-FASD "warped").
- **Replay**: **varie o brilho da tela** (baixo/médio/alto). Se todo replay for tela
  no máximo, o modelo cola "muito brilhante = fake". Varie **distância e ângulo**
  tela-câmera para o moiré aparecer em intensidades diferentes (às vezes forte, às
  vezes ausente) — para o modelo não depender só de moiré.
- **Moldura/bezel e mão** — ver checklist anti-viés (seção 8), é o ponto mais
  sensível por causa do crop 2.7× do MiniFASNet.

---

## 4. Parâmetros da câmera

| Parâmetro | Escolha | Motivo |
|---|---|---|
| Resolução | **1280×720 nativa** | Sem downscale interno; folga para o crop |
| Frame rate | **30 fps** contínuo | Padrão; amostramos frames esparsos depois |
| Foco | **FIXO** (travar, sem autofoco) | Autofoco borra frames e "respira" → pista espúria |
| Exposição/WB | **Auto ligado**, mas mesma política p/ real e ataque | Reflete uso real; fixar global impede generalização |
| Formato | **PNG** ou **JPEG q≥95**, **igual p/ todas as classes** | Pistas de PAD (moiré, textura, Fourier) vivem em alta frequência que compressão destrói |

> **Vazamento sutil e comum:** se reais forem PNG e ataques JPEG (ou resoluções
> diferentes), o modelo aprende artefato de compressão, não spoof. **Mesmo formato,
> mesma qualidade, mesma resolução para todas as classes.**

**Tamanho mínimo do rosto:** garantir bbox de face com **≥128 px** (idealmente
150–256) antes do crop. Como a entrada final é 80×80, precisamos de folga para o
downscale não destruir a textura. Em 720p, rosto ocupando ~1/5 a 1/2 da altura.

---

## 5. Quantidade alvo

Diversidade de **sujeitos** importa mais que frames por sujeito.

| Item | Mínimo | Ideal |
|---|---|---|
| Sujeitos | 20 | 40–50 (padrão CASIA/Replay) |
| Clips por sujeito | ~10 | 15–20 (cobrindo a matriz × PAIs) |
| Duração por clip | 10 s | 10–15 s (≈300–450 frames brutos) |
| **Frames úteis** por clip | ~10 | 10–30 (amostrados, não todos) |
| Frames úteis por sujeito | ~200 | 500–1000 (real+ataque somados) |

- **Não usar todos os frames.** Frames consecutivos são quase idênticos → inflam o
  dataset e as métricas. Amostrar **1 frame a cada ~0,5–1 s** (5–25 por clip),
  filtrando por: face detectada, nitidez (rejeitar blur), tamanho mínimo.
- **Balanceamento:** mirar ~1:1 real/fake **no treino** (ponderar/subamostrar, não
  duplicar). O **teste reflete a proporção real** — ou, melhor, reportar APCER/BPCER
  que são independentes de prevalência.
- **Diversidade de sujeitos:** tons de pele, idade, óculos, barba, cabelo.

---

## 6. Organização — nomenclatura + manifest

Abordagem híbrida dos datasets sérios: **nome de arquivo autodescritivo** (inspeção
humana, à prova de re-split) **+ manifest CSV canônico** (fonte da verdade para o
código).

### Nomenclatura de arquivo

```
S{sujeito:03d}_L{ilum}_SE{sessao:02d}_A{pai}_c{clip:03d}_f{frame:05d}.png
```
Exemplo: `S007_Lbacklight_SE02_Aprint-glossy_c003_f00042.png`
(`A` = `live` para bona fide.)

### Estrutura de pastas — por SUJEITO, não por real/fake

Agrupar por sujeito/clip é o que protege contra vazamento. **Não** organizar em
`real/` vs `fake/` (isso força uma hierarquia e dificulta o split subject-disjoint).

```
datasets/
├── raw/                         ← vídeos/frames brutos por sujeito
│   └── S007/SE02/c003/ ...
├── manifest_v1.csv              ← fonte canônica (1 linha por frame útil)
├── splits/
│   └── folds_v1.json            ← sujeitos por dobra (versionado no git)
└── RGB_Images/                  ← patches gerados p/ o treino (formato do repo)
    ├── org_1_80x60/{0,1}/
    ├── 2.7_80x80/{0,1}/
    └── 4_80x80/{0,1}/
```

O `RGB_Images/` é derivado — gerado pelo `prepare_dataset.py` a partir do `raw/` +
`manifest`. O `raw/` + `manifest` + `splits/` são a fonte; o `RGB_Images/` é
descartável/reprodutível.

### Manifest CSV — colunas mínimas

```
filepath, subject_id, label, pai_type, illumination, session,
clip_id, frame_idx, split, fold, capture_date, dataset_version
```

- `label`: **0 = live, 1 = spoof** (alvo binário de treino).
- `pai_type`: `live | print-matte | print-glossy | replay-phone | replay-monitor`
  — **metadado**, não alvo de treino; é o que permite **APCER por tipo de ataque**.
- Por que CSV além de pastas: permite cortar por sujeito/PAI/iluminação sem mover
  arquivos, é versionável no git, e evita re-split acidental.

---

## 7. Split, rotulagem e avaliação

### Split — subject-disjoint, sempre

Três níveis de vazamento a bloquear, do mais óbvio ao mais sutil:
1. **Frame-level** — frames do mesmo clip em train e test (quase idênticos).
2. **Clip-level** — clips do mesmo sujeito compartilham fundo/roupa/luz.
3. **Identity-level** — o mais sutil: o mesmo rosto como referência de aprendizado.

**Split primeiro por `subject_id`** — todos os clips e frames (reais e ataques)
daquele sujeito vão juntos para o mesmo lado.

**Com poucos sujeitos**, um split fixo único tem variância enorme. Usar:
- **LOSO (Leave-One-Subject-Out)** ou **k-fold por sujeito**: treina em N−1 grupos,
  testa no que sobra, repete. Reportar **média ± desvio-padrão** — o desvio é tão
  importante quanto a média aqui.
- **Nested CV** para escolher threshold/early-stopping: dentro de cada dobra de
  treino, separar 1–2 sujeitos para validação. **Nunca** escolher threshold no fold
  de teste.
- Reservar, se possível, **um PAI ou variação "não vista"** para teste (imita OULU-P2
  / SiW-P3): mede detecção de ataque novo, não memorização.

### Rotulagem — binária para treinar, fina para avaliar

- **Treino:** binário `live/spoof`. Multiclasse por tipo de ataque **fragmenta** o
  dataset pequeno e tende a overfitar no PAI → ficamos no binário.
- **Avaliação:** usar o `pai_type` (metadado) para reportar **APCER por espécie**.

### Métricas (ISO/IEC 30107-3)

```
APCER = fração de ATAQUES aceitos como real  →  reportar o PIOR PAI: max_i APCER_i
BPCER = fração de REAIS rejeitados como fake
ACER  = (APCER + BPCER) / 2   (reportar sempre o par, não só o ACER)
```

- **Threshold escolhido na validação** (ponto de EER do dev), depois **fixo** no
  teste. Nunca calibrar no teste.
- Reportar também **BPCER @ APCER fixo** (ex.: @1% e @5%) e a **curva DET/ROC** — é o
  que permite re-sintonizar o limiar em produção sem retreinar. Substitui o `0.7`
  arbitrário atual do nosso `predict.py`/`webcam.py`.
- **Reprodutibilidade:** fixar seed, registrar hash do manifest, versão do dataset e
  git-commit em cada experimento.

---

## 8. Checklist anti-viés (revisar antes de cada sessão)

- [ ] A iluminação desta sessão vai aparecer **também** em real E em ataque?
- [ ] O fundo desta sessão terá **real E ataque**? (nunca "real=sala A, fake=sala B")
- [ ] Mesmo **formato/qualidade/resolução** de arquivo para todas as classes?
- [ ] A **mão** aparece só nos ataques? → incluir ataques em **suporte fixo/tripé** e
      incluir **mão/movimento também nos reais**.
- [ ] O **bezel da tela / borda do papel** está sempre visível? → às vezes preencher
      o quadro (borda fora do crop 2.7×), às vezes deixar visível.
- [ ] O **moiré** é o único diferencial dos replays? → variar densidade/distância.
- [ ] O mesmo sujeito **não** está em dois splits? (garantido pelo split por sujeito)
- [ ] Cada sujeito tem **live E todos os PAIs** que quero avaliar? (senão o LOSO
      remove um PAI inteiro do treino ao tirar o sujeito)

---

## 9. Roteiro operacional de uma sessão

Para cada sujeito, uma sessão cobre:

1. **Consentimento** (LGPD) — registrar; usar ID anônimo, nunca nome no arquivo.
2. **4 clips reais** — um por condição de iluminação (frontal, lateral, contraluz,
   penumbra), variando pose/distância/expressão dentro do clip, trocando o fundo
   entre clips.
3. **Fotografar o sujeito** em alta qualidade para fabricar os PAIs (mesma
   identidade em real e ataque → modelo aprende textura, não identidade).
4. **Clips de ataque** — para cada PAI (print fosco, print brilhante, replay phone,
   replay monitor), repetir as condições de iluminação, variando reflexo/brilho e
   enquadramento (bezel dentro/fora), alternando mão e suporte.
5. Conferir o **checklist anti-viés** (seção 8) antes de encerrar.

Meta por sessão: ~10–20 clips, ~200–1000 frames úteis após amostragem.

---

## 10. Próximos passos (código)

- [ ] `collect_data.py` — guia a sessão pela webcam já configurada: pede sujeito,
      iluminação, PAI; grava clips nomeados na convenção da seção 6; escreve o
      manifest incrementalmente.
- [ ] `prepare_dataset.py` — extrai frames úteis (amostragem temporal + filtro de
      nitidez/tamanho), detecta face, gera patches multi-escala em `RGB_Images/`,
      e produz o **split subject-disjoint** (`folds_v1.json`).
- [ ] Data card do dataset (escopo "1 webcam, N sujeitos", demografia agregada,
      PAIs, limitações).

---

## Fontes

- OULU-NPU (4 protocolos, split 20/15/20 subject-disjoint): https://scispace.com/pdf/oulu-npu-a-mobile-face-presentation-attack-database-with-4x8pmjwi5n.pdf
- SiW / Liu et al. CVPR 2018 (protocolos pose, cross-medium, cross-PAI): https://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Learning_Deep_Models_CVPR_2018_paper.pdf
- CelebA-Spoof (manifest com 43 atributos): https://arxiv.org/pdf/2007.12342
- ISO/IEC 30107-3 (APCER/BPCER/ACER, APCER=max sobre PAIs): https://www.christoph-busch.de/files/Busch-PAD-240701.pdf
- Seleção de threshold (EER no dev, BPCER@APCER): https://ceur-ws.org/Vol-3742/short2.pdf
- Métricas ChaLearn LAP: https://chalearnlap.cvc.uab.cat/challenge/33/track/33/metrics/
