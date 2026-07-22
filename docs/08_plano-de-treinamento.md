# Plano de treinamento — MiniFASNet com dataset próprio

## Por que treinar com câmera própria

O modelo original foi treinado pela Minivision em câmeras Android específicas.
Cada câmera tem características que afetam a textura capturada pela rede:

| Fator | Impacto |
|---|---|
| Resposta de cor (ISP) | Saturação e balanço de branco diferentes |
| Compressão de vídeo | Artefatos JPEG/H264 específicos por fabricante |
| Foco e nitidez | Sharpening aplicado pelo driver |
| Ruído | Padrão de noise único do sensor |
| Distorção de lente | Barrel/pincushion distortion |

A rede aprende a distinguir "pele real" de "pele em papel/tela" através dessas texturas.
Mudar a câmera muda as texturas — a fronteira de decisão fica errada.

---

## Estratégia recomendada: fine-tuning

Treinar do zero exige ~10k+ amostras. Fine-tuning a partir dos pesos existentes
converge com muito menos dados porque o backbone já aprendeu features de face.

```
Backbone (camadas convolucionais) → congelar parcialmente
Cabeça de classificação (linear + bn + prob) → retreinar
```

Referência de dados mínimos por abordagem:

| Abordagem | Real | Fake | Tempo estimado (CPU) |
|---|---|---|---|
| Só cabeça (head-only) | 200+ | 200+ | 1-2h |
| Últimas 2 stages | 500+ | 500+ | 4-8h |
| Full fine-tuning | 1000+ | 1000+ | 1-2 dias |
| Do zero | 5000+ | 5000+ | dias/semanas |

---

## Tipos de ataque a cobrir

Para cada tipo de fake o modelo aprende texturas diferentes.
Cobrir mais tipos = modelo mais robusto.

| Tipo | Dificuldade de coleta | Prioridade |
|---|---|---|
| Foto impressa (papel fosco) | Baixa | Alta |
| Foto impressa (papel brilhante) | Baixa | Alta |
| Replay em tela de celular | Baixa | Alta |
| Replay em monitor | Baixa | Média |
| Replay em tablet | Baixa | Média |
| Máscara 2D (papel recortado) | Média | Média |
| Máscara 3D (silicone) | Alta | Baixa |

---

## Estrutura do dataset

```
datasets/
├── train/
│   ├── real/          ← vídeos/frames de rostos reais
│   └── fake/          ← vídeos/frames de ataques
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

Split sugerido: **70% treino / 15% val / 15% teste**

Cada entrada pode ser um frame ou um vídeo curto.
O treino usa frames extraídos dos vídeos + detecção de face para gerar os patches.

---

## Protocolo de coleta

### Real (rosto ao vivo)
- Gravar com a câmera alvo (a mesma que será usada em produção)
- Mínimo 5 pessoas diferentes
- Variações por pessoa:
  - Iluminação: ambiente, lateral, contra-luz suave
  - Ângulo: frontal, ±15°, ±30° (horizontal e vertical)
  - Distância: 30cm, 60cm, 100cm
  - Expressões: neutro, sorrindo, falando
- Duração: 10-15 segundos por clip (→ ~150-225 frames a 15fps)

### Fake (ataques)
- Usar as **mesmas fotos das pessoas reais** para criar os ataques
  (isso garante que o modelo não memorize identidade, só textura)
- Print attack: imprimir em A4, segurar na frente da câmera
- Replay attack: exibir no celular/monitor, gravar com a câmera alvo
- Variar distância e ângulo igual ao real

### Balanceamento
- Manter proporção 1:1 real/fake no treino
- Se tiver mais fake que real, subsamplear fake

---

## Estrutura de dataset exigida pelo repo original

O treino usa **múltiplas versões do mesmo frame** — a imagem original redimensionada
e patches recortados em torno do rosto em diferentes escalas.
Cada combinação scale+tamanho corresponde a um modelo diferente no ensemble.

```
datasets/
└── RGB_Images/
    ├── org_1_80x60/        ← imagem original redimensionada (sem crop de face)
    │   ├── 0/              ← classe 0: fake
    │   ├── 1/              ← classe 1: real
    │   └── 2/              ← classe 2: auxiliar (Fourier supervision)
    ├── 1_80x80/            ← patch scale=1.0
    │   ├── 0/
    │   ├── 1/
    │   └── 2/
    ├── 2.7_80x80/          ← patch scale=2.7  (modelo MiniFASNetV2)
    │   ├── 0/
    │   ├── 1/
    │   └── 2/
    └── 4_80x80/            ← patch scale=4.0  (modelo MiniFASNetV1SE)
        ├── 0/
        ├── 1/
        └── 2/
```

**Classe 2** não é "ignorada" — é usada na supervisão auxiliar via espectro de Fourier
(gerado online durante o treino). É por isso que os modelos têm `num_classes=3`.

O nome das pastas segue a mesma convenção dos arquivos `.pth`:
`<scale>_<h>x<w>` → `2.7_80x80` → `2.7_80x80_MiniFASNetV2.pth`.

---

## Pipeline de preparação de dados (segundo o README original)

```
frames brutos (câmera alvo)
    ↓
detecção de face — RetinaFace (insightface) ou Caffe bundled
    ↓  para cada frame:
    ├── org: resize para 80x60 → salvar em org_1_80x60/<classe>/
    ├── patch scale=1.0: crop face + resize 80x80 → 1_80x80/<classe>/
    ├── patch scale=2.7: crop face expandida 2.7x + resize 80x80 → 2.7_80x80/<classe>/
    └── patch scale=4.0: crop face expandida 4.0x + resize 80x80 → 4_80x80/<classe>/
```

O script `generate_patches.py` (já presente em nosso projeto como `liveness/cropper.py`)
implementa o crop com escala. Precisaremos de um script que itere frames,
detecte faces e salve todas as versões na estrutura acima.

**Detector recomendado pelo repo:** RetinaFace (insightface)
`https://github.com/deepinsight/insightface/tree/master/RetinaFace`

O detector Caffe bundled (`resources/detection/`) pode ser usado como alternativa
mais simples, sem precisar instalar insightface.

---

## Configuração de treinamento

Baseado no `default_config.py` original, com ajustes para fine-tuning:

```python
# Fine-tuning (sugerido)
lr           = 1e-3        # menor que o original (1e-1) para não destruir os pesos
milestones   = [5, 10, 15] # decaimento mais cedo
epochs       = 20          # suficiente para fine-tuning
batch_size   = 64          # reduzir se memória limitada
momentum     = 0.9
num_classes  = 3           # manter 3 — classe 2 é usada na supervisão Fourier
embedding_size = 128

# Comando original:
# python train.py --device_ids 0 --patch_info 2.7_80x80
```

Treinar um modelo por patch_info (`2.7_80x80` e `4_0_0_80x80` separadamente).

---

## Métricas de avaliação

A métrica padrão em liveness é **ACER** (Average Classification Error Rate):

```
APCER = taxa de ataques aceitos como real  (Attack Presentation Classification Error Rate)
BPCER = taxa de reais rejeitados como fake (Bona fide Presentation Classification Error Rate)
ACER  = (APCER + BPCER) / 2
```

Performance reportada pelo repo no modelo open-source APK:
- FPR = 1e-5 → TPR = 97.8%

Meta razoável para começar com dataset próprio: **ACER < 0.05** (5%).

Também monitorar:
- Curva ROC + AUC
- Score por tipo de ataque (paper vs replay vs etc.)

---

## Próximos passos (ordem)

- [ ] 1. Criar script de coleta de vídeos via webcam (`collect_data.py`)
- [ ] 2. Criar script de extração de frames e geração de patches (`prepare_dataset.py`)
- [ ] 3. Adaptar `train.py` e `train_main.py` do repo original para Python 3.10+
- [ ] 4. Treinar os dois modelos (`2.7_80x80` e `4_0_0_80x80`) com fine-tuning
- [ ] 5. Avaliar ACER/APCER/BPCER no conjunto de test
- [ ] 6. Exportar os novos pesos para ONNX (`convert_to_onnx.py` já pronto)
- [ ] 7. Testar no `webcam.py` com os novos modelos

---

## Referências

- README original: `../Silent-Face-Anti-Spoofing/README_EN.md`
- Detector RetinaFace (insightface): https://github.com/deepinsight/insightface
- ISO/IEC 30107-3: Presentation Attack Detection — define APCER/BPCER/ACER
