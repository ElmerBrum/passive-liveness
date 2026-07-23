"""
Prepara o CelebA-Spoof para o formato de patches que o dataset.py/train.py consomem.

Estrutura de saída (tem que bater EXATAMENTE com o contrato do config.py/README):

    <out>/<patch_info>/<split>/<class>/*.png     ex.: training/data/2.7_80x80/train/1/img_000123.png
    <out>/manifest.csv   colunas: filepath, subject_id, label, pai_type, split

    split : train | val | test
    class : 0 = spoof/fake , 1 = live/real
    imagem: 80x80, 3 canais, salva com cv2.imwrite (BGR), pixels [0,255]

Regras de corretude (ver docs/10_plano-dataset-publico.md, seções 5 e 6):

 1. INVERSÃO DO SINAL DO RÓTULO. No CelebA-Spoof, o campo `spoof_type`
    (índice 40 do vetor de anotação) usa 0 = Live/real; 1..10 = ataques.
    Aqui, pasta/rótulo 1 = real. Logo invertemos:
        spoof_type == 0     -> classe 1 (live)
        spoof_type in 1..10 -> classe 0 (spoof)
    O pai_type registra "live" (0) ou o nome grosseiro do ataque (1..10).

 2. BBOX EM ESCALA 224. Os arquivos *_BB.txt guardam "x y w h score" relativos
    a uma imagem 224x224. É OBRIGATÓRIO reescalar para o tamanho real da imagem:
        x_real = x * (W_real / 224)   (idem y/w/h com H_real/224)
    O último número da linha é um score e é descartado.

 3. CROP. Expande a bbox pelo fator de escala (padrão 2.7) em torno do centro,
    faz um crop QUADRADO, faz clamp nos limites da imagem e redimensiona para
    80x80. Reutiliza liveness/cropper.py::CropImage.crop. Se a bbox estiver
    faltando/inválida, a imagem é PULADA (não gravamos crop preto).

 4. BGR em todo o pipeline (cv2.imread dá BGR; cv2.imwrite espera BGR).
    NUNCA converter para RGB.

 5. SPLIT SUBJECT-DISJOINT. O CelebA-Spoof agrupa imagens por identidade
    (Data/train/<subject_id>/... e Data/test/<subject_id>/...). Dividimos por
    SUJEITO, nunca por imagem, com seed fixa. Todas as imagens de um sujeito
    (live e spoof) vão para o mesmo split.

Este script é defensivo quanto ao layout do CelebA-Spoof (varia entre espelhos):
localiza imagens caminhando atrás de *.png/*.jpg que tenham um *_BB.txt irmão e
lê o rótulo da anotação pareada; se houver um label json/txt conhecido, prefere-o.

Auto-teste: rode com --self-test para fabricar 2 "sujeitos" sintéticos com
imagens e _BB.txt dummy num diretório temporário, rodar o pipeline e verificar
a estrutura de saída e o mapeamento de rótulos. Requer cv2/numpy (pula com aviso
se não estiverem disponíveis).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# Import robusto do CropImage: funciona tanto rodando como módulo
# (python -m training.prepare_public_dataset) quanto como script solto.
try:
    from liveness.cropper import CropImage
except Exception:  # pragma: no cover - fallback de path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from liveness.cropper import CropImage


# ---------------------------------------------------------------------------
# Constantes do CelebA-Spoof
# ---------------------------------------------------------------------------

# As bboxes dos *_BB.txt são relativas a uma imagem 224x224 (armadilha #1).
BB_REFERENCE_SIZE = 224.0

# Índice do campo spoof_type no vetor de anotação por-imagem do CelebA-Spoof.
# O vetor tem 43 posições; a posição 40 é o "spoof type" (0=live, 1..10=ataque).
SPOOF_TYPE_INDEX = 40

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

# Mapeamento grosseiro spoof_type -> nome de PAI (ver docs/10 seção 5):
#   0        = live
#   1..3     = print
#   4..6     = paper (paper-cut)
#   7..9     = replay
#   10       = mask (3D)
def spoof_type_to_pai(spoof_type: int) -> str:
    if spoof_type == 0:
        return "live"
    if 1 <= spoof_type <= 3:
        return "print"
    if 4 <= spoof_type <= 6:
        return "paper"
    if 7 <= spoof_type <= 9:
        return "replay"
    if spoof_type == 10:
        return "mask"
    return "unknown"


def spoof_type_to_label(spoof_type: int) -> int:
    """INVERSÃO DO SINAL: spoof_type 0 (live) -> 1 (real); 1..10 (ataque) -> 0."""
    return 1 if spoof_type == 0 else 0


# ---------------------------------------------------------------------------
# Estrutura de uma amostra descoberta
# ---------------------------------------------------------------------------

class Sample:
    """Uma imagem candidata com sua anotação pareada."""

    __slots__ = ("img_path", "bb_path", "subject_id", "spoof_type")

    def __init__(self, img_path: Path, bb_path: Path | None,
                 subject_id: str, spoof_type: int):
        self.img_path = img_path
        self.bb_path = bb_path
        self.subject_id = subject_id
        self.spoof_type = spoof_type

    @property
    def label(self) -> int:
        return spoof_type_to_label(self.spoof_type)

    @property
    def pai_type(self) -> str:
        return spoof_type_to_pai(self.spoof_type)


# ---------------------------------------------------------------------------
# Parsing de anotações (defensivo — layout varia entre espelhos)
# ---------------------------------------------------------------------------

def _load_label_jsons(celeba_root: Path) -> dict[str, int]:
    """
    Tenta carregar os label jsons conhecidos do CelebA-Spoof
    (ex.: metas/intra_test/train_label.json, test_label.json).

    O formato desses jsons é { "<caminho relativo da imagem>": [43 ints] },
    onde a posição SPOOF_TYPE_INDEX guarda o spoof_type. Alguns espelhos
    guardam apenas o rótulo binário (0/1) como int simples; tratamos os dois.

    Retorna um dict mapeando o basename da imagem (e também o caminho relativo
    normalizado) -> spoof_type. Falha silenciosamente (dict vazio) se nada for
    encontrado.
    """
    labels: dict[str, int] = {}
    candidates = list(celeba_root.rglob("*_label.json"))
    # Também aceita nomes comuns tipo metas/intra_test/*.json
    candidates += [p for p in celeba_root.rglob("*.json")
                   if "label" in p.name.lower() and p not in candidates]

    for jf in candidates:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
        except Exception as e:  # pragma: no cover - json corrompido
            print(f"[aviso] falha lendo {jf}: {e}")
            continue
        if not isinstance(data, dict):
            continue
        for rel, ann in data.items():
            spoof_type = _extract_spoof_type(ann)
            if spoof_type is None:
                continue
            key_full = _normalize_key(rel)
            key_base = Path(rel).name
            labels[key_full] = spoof_type
            # basename só é usado como fallback; não sobrescreve se colidir
            labels.setdefault(key_base, spoof_type)

    if labels:
        print(f"[info] carregados {len(labels)} rótulos de label json(s)")
    return labels


def _extract_spoof_type(ann) -> int | None:
    """Extrai spoof_type de uma anotação que pode ser lista(43) ou int simples."""
    if isinstance(ann, list):
        if len(ann) > SPOOF_TYPE_INDEX:
            try:
                return int(ann[SPOOF_TYPE_INDEX])
            except (TypeError, ValueError):
                return None
        return None
    if isinstance(ann, (int, float)):
        # Rótulo binário simples. Convenção do CelebA-Spoof: 0 = live.
        # Como só temos 0/1, mapeamos 1 -> spoof_type 1 (print genérico).
        return 0 if int(ann) == 0 else 1
    return None


def _normalize_key(rel: str) -> str:
    """Normaliza um caminho relativo para casar com os caminhos descobertos."""
    return str(Path(rel)).replace("\\", "/")


def _parse_spoof_type_from_txt(txt_path: Path) -> int | None:
    """
    Alguns espelhos guardam, ao lado da imagem, um *.txt (não o _BB.txt) com o
    vetor de anotação. Tenta ler spoof_type dele.
    """
    try:
        with open(txt_path, "r") as f:
            tokens = f.read().split()
    except Exception:
        return None
    if len(tokens) > SPOOF_TYPE_INDEX:
        try:
            return int(float(tokens[SPOOF_TYPE_INDEX]))
        except ValueError:
            return None
    return None


def _spoof_type_from_path(img_path: Path) -> int | None:
    """
    Fallback final: infere live/spoof pela estrutura de diretórios.
    O CelebA-Spoof organiza como .../<subject>/live/... e .../<subject>/spoof/...
    Retorna 0 (live) ou 1 (spoof genérico) ou None se indeterminado.
    """
    parts_lower = [p.lower() for p in img_path.parts]
    if "live" in parts_lower:
        return 0
    if "spoof" in parts_lower:
        return 1
    return None


def _subject_id_from_path(img_path: Path, celeba_root: Path) -> str:
    """
    Deriva o subject_id da estrutura de diretórios.
    Layout típico: <root>/Data/<split>/<subject_id>/{live,spoof}/<img>
    Estratégia defensiva: procura o componente 'Data' e pega o diretório dois
    níveis abaixo (split/subject); se não achar, usa o diretório-pai imediato
    que não seja live/spoof.
    """
    try:
        rel = img_path.relative_to(celeba_root)
        parts = rel.parts
    except ValueError:
        parts = img_path.parts

    lower = [p.lower() for p in parts]
    # Caso Data/<split>/<subject>/...
    if "data" in lower:
        i = lower.index("data")
        # parts[i+1] = split (train/test), parts[i+2] = subject
        if len(parts) > i + 2:
            return parts[i + 2]
        if len(parts) > i + 1:
            return parts[i + 1]

    # Fallback: sobe a árvore ignorando pastas live/spoof e o próprio arquivo.
    for parent in img_path.parents:
        name = parent.name.lower()
        if name in ("live", "spoof", ""):
            continue
        return parent.name
    return "unknown_subject"


# ---------------------------------------------------------------------------
# Descoberta de amostras
# ---------------------------------------------------------------------------

def discover_samples(celeba_root: Path) -> list[Sample]:
    """
    Caminha pelo celeba_root procurando imagens que tenham um *_BB.txt irmão.
    Para cada uma, resolve o spoof_type na ordem de preferência:
        1. label json conhecido (se presente)
        2. *.txt de anotação irmão (vetor de 43)
        3. estrutura de diretórios (live/spoof)
    Imagens sem bbox são descobertas mesmo assim (bb_path=None) e serão puladas
    no crop com aviso — mas normalmente exigimos o _BB.txt irmão.
    """
    celeba_root = celeba_root.resolve()
    label_map = _load_label_jsons(celeba_root)

    samples: list[Sample] = []
    n_no_bb = 0
    n_no_label = 0

    for img_path in _iter_images(celeba_root):
        bb_path = img_path.with_name(img_path.stem + "_BB.txt")
        if not bb_path.is_file():
            # Sem bbox pareada: sem crop confiável. Contabiliza e pula.
            n_no_bb += 1
            continue

        spoof_type = _resolve_spoof_type(img_path, celeba_root, label_map)
        if spoof_type is None:
            n_no_label += 1
            continue

        subject_id = _subject_id_from_path(img_path, celeba_root)
        samples.append(Sample(img_path, bb_path, subject_id, spoof_type))

    if n_no_bb:
        print(f"[aviso] {n_no_bb} imagens sem _BB.txt irmão foram ignoradas")
    if n_no_label:
        print(f"[aviso] {n_no_label} imagens sem rótulo resolvível foram ignoradas")
    print(f"[info] {len(samples)} amostras válidas descobertas")
    return samples


def _iter_images(root: Path):
    """Gera todos os arquivos de imagem sob root (ignorando os _BB.txt)."""
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def _resolve_spoof_type(img_path: Path, celeba_root: Path,
                        label_map: dict[str, int]) -> int | None:
    """Resolve spoof_type na ordem de preferência descrita em discover_samples."""
    # 1. label json (por caminho relativo normalizado, depois por basename)
    if label_map:
        try:
            rel_key = _normalize_key(str(img_path.relative_to(celeba_root)))
        except ValueError:
            rel_key = _normalize_key(str(img_path))
        if rel_key in label_map:
            return label_map[rel_key]
        if img_path.name in label_map:
            return label_map[img_path.name]

    # 2. *.txt de anotação irmão (mesmo stem, sem _BB)
    ann_txt = img_path.with_suffix(".txt")
    if ann_txt.is_file() and ann_txt.name != (img_path.stem + "_BB.txt"):
        st = _parse_spoof_type_from_txt(ann_txt)
        if st is not None:
            return st

    # 3. estrutura de diretórios
    return _spoof_type_from_path(img_path)


# ---------------------------------------------------------------------------
# Parsing e reescala da bbox
# ---------------------------------------------------------------------------

def parse_bb(bb_path: Path, real_w: int, real_h: int) -> list[float] | None:
    """
    Lê "x y w h score" (relativo a 224x224) e reescala para o tamanho real.
    Retorna [x, y, w, h] em pixels reais, ou None se inválido.
    """
    try:
        with open(bb_path, "r") as f:
            tokens = f.read().split()
    except Exception as e:
        print(f"[aviso] falha lendo bbox {bb_path}: {e}")
        return None

    if len(tokens) < 4:
        print(f"[aviso] bbox malformada (poucos campos): {bb_path}")
        return None

    try:
        x, y, w, h = (float(tokens[0]), float(tokens[1]),
                      float(tokens[2]), float(tokens[3]))
    except ValueError:
        print(f"[aviso] bbox não-numérica: {bb_path}")
        return None

    # Reescala 224 -> tamanho real (armadilha #1).
    sx = real_w / BB_REFERENCE_SIZE
    sy = real_h / BB_REFERENCE_SIZE
    x, y, w, h = x * sx, y * sy, w * sx, h * sy

    if w <= 1 or h <= 1:
        print(f"[aviso] bbox degenerada (w={w:.1f}, h={h:.1f}): {bb_path}")
        return None

    return [x, y, w, h]


# ---------------------------------------------------------------------------
# Split subject-disjoint
# ---------------------------------------------------------------------------

def split_subjects(subject_ids: list[str], val_frac: float, test_frac: float,
                   seed: int) -> dict[str, str]:
    """
    Divide sujeitos (não imagens) em train/val/test com seed fixa.
    Retorna dict subject_id -> split.
    """
    uniq = sorted(set(subject_ids))
    rng = random.Random(seed)
    rng.shuffle(uniq)

    n = len(uniq)
    n_val = int(round(n * val_frac))
    n_test = int(round(n * test_frac))
    # garante que train não fique vazio quando há sujeitos suficientes
    n_val = min(n_val, max(0, n - 1))
    n_test = min(n_test, max(0, n - 1 - n_val))

    val_ids = set(uniq[:n_val])
    test_ids = set(uniq[n_val:n_val + n_test])

    mapping: dict[str, str] = {}
    for sid in uniq:
        if sid in val_ids:
            mapping[sid] = "val"
        elif sid in test_ids:
            mapping[sid] = "test"
        else:
            mapping[sid] = "train"
    return mapping


def subsample_per_subject(samples: list[Sample], max_per_subject: int | None,
                          seed: int) -> list[Sample]:
    """Limita o nº de frames por sujeito (opcional), preservando classes."""
    if not max_per_subject or max_per_subject <= 0:
        return samples
    by_subject: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_subject[s.subject_id].append(s)

    rng = random.Random(seed)
    out: list[Sample] = []
    for sid, group in by_subject.items():
        if len(group) <= max_per_subject:
            out.extend(group)
        else:
            out.extend(rng.sample(group, max_per_subject))
    return out


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def prepare(celeba_root: Path, out_dir: Path, patch_info: str,
            bbox_scale: float, size: int, val_frac: float, test_frac: float,
            max_per_subject: int | None, seed: int) -> dict:
    """
    Executa a preparação completa. Retorna um dict de resumo (contagens).
    Importa cv2/numpy aqui dentro para permitir o self-test pular graciosamente
    quando não estiverem instalados.
    """
    import cv2  # noqa: F401  (import tardio proposital)

    cropper = CropImage()

    samples = discover_samples(celeba_root)
    if not samples:
        print("[erro] nenhuma amostra válida encontrada. Verifique --celeba-root.")
        return {"written": 0}

    samples = subsample_per_subject(samples, max_per_subject, seed)

    split_map = split_subjects([s.subject_id for s in samples],
                               val_frac, test_frac, seed)

    patch_root = out_dir / patch_info
    # cria a árvore <split>/<class>
    for split in ("train", "val", "test"):
        for cls in ("0", "1"):
            (patch_root / split / cls).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    # contadores[split][class] = int
    counts: dict[str, dict[int, int]] = {
        s: {0: 0, 1: 0} for s in ("train", "val", "test")
    }
    n_skipped = 0
    n_written = 0

    for idx, s in enumerate(samples):
        split = split_map[s.subject_id]
        label = s.label

        img = cv2.imread(str(s.img_path), cv2.IMREAD_COLOR)  # BGR uint8
        if img is None:
            print(f"[aviso] falha lendo imagem: {s.img_path}")
            n_skipped += 1
            continue

        real_h, real_w = img.shape[:2]
        bbox = parse_bb(s.bb_path, real_w, real_h) if s.bb_path else None
        if bbox is None:
            n_skipped += 1
            continue

        # Crop quadrado expandido por bbox_scale, com clamp, resize size x size.
        # CropImage.crop faz clamp interno e usa max side implicitamente via
        # _get_new_box (escala em torno do centro). Garantimos crop quadrado
        # usando um lado = max(w, h) para a bbox base.
        x, y, w, h = bbox
        side = max(w, h)
        cx, cy = x + w / 2.0, y + h / 2.0
        square_bbox = [cx - side / 2.0, cy - side / 2.0, side, side]

        try:
            patch = cropper.crop(
                org_img=img, bbox=square_bbox, scale=bbox_scale,
                out_w=size, out_h=size, crop=True,
            )
        except Exception as e:
            print(f"[aviso] crop falhou em {s.img_path}: {e}")
            n_skipped += 1
            continue

        if patch is None or patch.size == 0:
            n_skipped += 1
            continue

        # Nome de arquivo estável e único.
        fname = f"img_{idx:07d}.png"
        rel_out = Path(patch_info) / split / str(label) / fname
        abs_out = out_dir / rel_out

        # cv2.imwrite espera BGR — mantemos BGR (armadilha #2).
        ok = cv2.imwrite(str(abs_out), patch)
        if not ok:
            print(f"[aviso] cv2.imwrite falhou: {abs_out}")
            n_skipped += 1
            continue

        counts[split][label] += 1
        n_written += 1
        manifest_rows.append({
            "filepath": str(rel_out).replace("\\", "/"),
            "subject_id": s.subject_id,
            "label": label,
            "pai_type": s.pai_type,
            "split": split,
        })

    # Escreve o manifest.csv.
    manifest_path = out_dir / "manifest.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filepath", "subject_id", "label", "pai_type", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Resumo.
    _print_summary(counts, n_written, n_skipped, split_map, manifest_path)

    return {
        "written": n_written,
        "skipped": n_skipped,
        "counts": counts,
        "manifest": str(manifest_path),
        "n_subjects": len(set(split_map)),
    }


def _print_summary(counts, n_written, n_skipped, split_map, manifest_path):
    print("\n" + "=" * 60)
    print("RESUMO DA PREPARAÇÃO")
    print("=" * 60)
    # sujeitos por split
    subj_by_split: dict[str, set] = defaultdict(set)
    for sid, sp in split_map.items():
        subj_by_split[sp].add(sid)
    for split in ("train", "val", "test"):
        c = counts[split]
        print(f"  {split:5s} | sujeitos={len(subj_by_split[split]):5d} "
              f"| spoof(0)={c[0]:7d} | live(1)={c[1]:7d} "
              f"| total={c[0] + c[1]:7d}")
    print("-" * 60)
    print(f"  imagens gravadas : {n_written}")
    print(f"  imagens puladas  : {n_skipped}")
    print(f"  manifest         : {manifest_path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Auto-teste sintético
# ---------------------------------------------------------------------------

def run_self_test() -> int:
    """
    Fabrica 2 sujeitos sintéticos (cada um com 1 imagem live + 1 spoof) num
    diretório temporário, com _BB.txt válidos, roda o pipeline e verifica:
      - a estrutura de saída <patch_info>/<split>/<class>/*.png
      - o mapeamento de rótulos (spoof_type 0 -> pasta 1; 1..10 -> pasta 0)
      - o manifest.csv com as colunas certas
      - split subject-disjoint (um sujeito não aparece em dois splits)
    Pula graciosamente se cv2/numpy não estiverem disponíveis.
    """
    try:
        import cv2
        import numpy as np
    except Exception as e:
        print(f"[self-test] PULADO: cv2/numpy indisponível ({e})")
        return 0

    print("[self-test] fabricando dataset sintético...")
    tmp = Path(tempfile.mkdtemp(prefix="celeba_selftest_"))
    celeba_root = tmp / "CelebA_Spoof"
    out_dir = tmp / "out"

    # Layout: Data/train/<subject>/{live,spoof}/<img> + _BB.txt
    # spoof_type embutido num *.txt de anotação irmão (43 tokens).
    def make_image(dir_path: Path, name: str, spoof_type: int,
                   real_w: int, real_h: int):
        dir_path.mkdir(parents=True, exist_ok=True)
        img_path = dir_path / f"{name}.png"
        # imagem colorida com um retângulo "rosto" no meio
        img = np.zeros((real_h, real_w, 3), dtype=np.uint8)
        img[:] = (30, 60, 90)  # BGR de fundo
        # rosto (bbox real que vamos codificar em escala 224)
        cv2.rectangle(img, (real_w // 4, real_h // 4),
                      (3 * real_w // 4, 3 * real_h // 4), (200, 180, 160), -1)
        cv2.imwrite(str(img_path), img)

        # bbox real -> converter para escala 224 para o _BB.txt
        bx = real_w // 4
        by = real_h // 4
        bw = real_w // 2
        bh = real_h // 2
        sx = BB_REFERENCE_SIZE / real_w
        sy = BB_REFERENCE_SIZE / real_h
        bb_line = f"{bx * sx:.1f} {by * sy:.1f} {bw * sx:.1f} {bh * sy:.1f} 0.99\n"
        (dir_path / f"{name}_BB.txt").write_text(bb_line)

        # anotação de 43 tokens com spoof_type na posição SPOOF_TYPE_INDEX
        ann = ["0"] * 43
        ann[SPOOF_TYPE_INDEX] = str(spoof_type)
        (dir_path / f"{name}.txt").write_text(" ".join(ann) + "\n")

    for subj in ("subjA", "subjB"):
        base = celeba_root / "Data" / "train" / subj
        # live (spoof_type 0) -> deve virar classe 1
        make_image(base / "live", "live_0", spoof_type=0,
                   real_w=200, real_h=300)
        # spoof print (spoof_type 1) -> deve virar classe 0
        make_image(base / "spoof", "spoof_1", spoof_type=1,
                   real_w=200, real_h=300)

    # Roda o pipeline. val_frac/test_frac altos p/ garantir sujeitos em splits
    # distintos com apenas 2 sujeitos (1 val, 1 train por arredondamento).
    summary = prepare(
        celeba_root=celeba_root, out_dir=out_dir, patch_info="2.7_80x80",
        bbox_scale=2.7, size=80, val_frac=0.5, test_frac=0.0,
        max_per_subject=None, seed=42,
    )

    # --- Verificações ---
    ok = True

    # 1. gravou 4 imagens (2 sujeitos x 2 imagens)
    if summary["written"] != 4:
        print(f"[self-test] FALHA: esperado 4 gravadas, obtido {summary['written']}")
        ok = False

    # 2. estrutura de pastas e tamanho/canais das imagens
    patch_root = out_dir / "2.7_80x80"
    found_pngs = list(patch_root.rglob("*.png"))
    if len(found_pngs) != 4:
        print(f"[self-test] FALHA: {len(found_pngs)} pngs no disco (esperado 4)")
        ok = False
    for p in found_pngs:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None or im.shape != (80, 80, 3):
            print(f"[self-test] FALHA: {p} não é 80x80x3 (shape={None if im is None else im.shape})")
            ok = False

    # 3. mapeamento de rótulos via manifest
    manifest = out_dir / "manifest.csv"
    with open(manifest) as f:
        rows = list(csv.DictReader(f))
    if [*rows[0].keys()] != ["filepath", "subject_id", "label", "pai_type", "split"]:
        print(f"[self-test] FALHA: colunas do manifest erradas: {list(rows[0].keys())}")
        ok = False

    # cada linha: se filepath está em /1/ então label==1 e pai_type==live;
    # se em /0/ então label==0 e pai_type != live
    for r in rows:
        parts = Path(r["filepath"]).parts
        cls_folder = parts[-2]
        if cls_folder != r["label"]:
            print(f"[self-test] FALHA: pasta {cls_folder} != label {r['label']}")
            ok = False
        if r["label"] == "1" and r["pai_type"] != "live":
            print(f"[self-test] FALHA: label 1 mas pai_type={r['pai_type']}")
            ok = False
        if r["label"] == "0" and r["pai_type"] == "live":
            print("[self-test] FALHA: label 0 mas pai_type=live")
            ok = False

    # 4. contagem: 2 live (classe 1) e 2 spoof (classe 0) no total
    n_live = sum(1 for r in rows if r["label"] == "1")
    n_spoof = sum(1 for r in rows if r["label"] == "0")
    if n_live != 2 or n_spoof != 2:
        print(f"[self-test] FALHA: live={n_live} spoof={n_spoof} (esperado 2/2)")
        ok = False

    # 5. subject-disjoint: nenhum subject_id em mais de um split
    subj_splits: dict[str, set] = defaultdict(set)
    for r in rows:
        subj_splits[r["subject_id"]].add(r["split"])
    for sid, splits in subj_splits.items():
        if len(splits) != 1:
            print(f"[self-test] FALHA: sujeito {sid} em múltiplos splits {splits}")
            ok = False

    if ok:
        print("[self-test] OK: estrutura, rótulos (invertidos), manifest e "
              "split subject-disjoint corretos.")
        return 0
    print("[self-test] FALHOU.")
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Converte CelebA-Spoof para patches 80x80 BGR + manifest.")
    p.add_argument("--celeba-root", type=Path,
                   help="Raiz do CelebA-Spoof (contém Data/ e metadados).")
    p.add_argument("--out", type=Path, default=Path("training/data"),
                   help="Diretório de saída (default: training/data).")
    p.add_argument("--patch-info", type=str, default="2.7_80x80",
                   help="Rótulo de escala/tamanho (default: 2.7_80x80).")
    p.add_argument("--bbox-scale", type=float, default=2.7,
                   help="Fator de expansão da bbox (default: 2.7).")
    p.add_argument("--size", type=int, default=80,
                   help="Lado do patch de saída em pixels (default: 80).")
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="Fração de SUJEITOS para validação (default: 0.15).")
    p.add_argument("--test-frac", type=float, default=0.15,
                   help="Fração de SUJEITOS para teste (default: 0.15).")
    p.add_argument("--max-per-subject", type=int, default=None,
                   help="Cap opcional de frames por sujeito (default: sem cap).")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed do split subject-disjoint (default: 42).")
    p.add_argument("--self-test", action="store_true",
                   help="Roda um auto-teste sintético e sai.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.self_test:
        return run_self_test()

    if args.celeba_root is None:
        print("[erro] --celeba-root é obrigatório (ou use --self-test).")
        return 2
    if not args.celeba_root.exists():
        print(f"[erro] --celeba-root não existe: {args.celeba_root}")
        return 2

    prepare(
        celeba_root=args.celeba_root,
        out_dir=args.out,
        patch_info=args.patch_info,
        bbox_scale=args.bbox_scale,
        size=args.size,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        max_per_subject=args.max_per_subject,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
