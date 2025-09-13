#!/usr/bin/env bash
set -euo pipefail # 오류·미정의 변수·파이프 실패 시 즉시 종료.
umask 077 # 생성 파일/디렉토리는 오너만 접근하기

PYTHON="/home/undergrad/.conda/envs/tc3/bin/python"   # ← 환경의 python 절대경로(현재 우리 환경 tc3로 고정)

JOB_JSON=""

# job json만 받아오기 누락이거나, 알 수 없는 인자일시 종료
while [[ $# -gt 0 ]]; do
  case "$1" in
    --job) JOB_JSON="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$JOB_JSON" ]]; then
  echo "Usage: $0 --job <job.json>" >&2
  exit 2
fi

# 스크립트 위치(프로젝트 루트)로 이동
cd "$(dirname "$0")"

# 1) 전처리: job.json -> out_root/<job_id>/dataset/{i_s/*.png, i_s.txt, i_t.txt}
# 즉, json파일에 담겨있는 원본이미지 + area뭉치들을 풀어서 ssh_input에 넣어놓겠다는 말
DATASET_DIR="$("$PYTHON" - "$JOB_JSON" <<'PY'
import sys, json, os
from PIL import Image

job_path = sys.argv[1]
with open(job_path, "r", encoding="utf-8") as f:
    job = json.load(f)

inp      = job["input_path"]
out_root = job["out_root"]
job_id   = job["job_id"]
digits   = int(job.get("naming", {}).get("digits", 5))
start    = int(job.get("naming", {}).get("start", 0))
areas    = job.get("areas", [])

dataset_dir = os.path.join(out_root, job_id, "dataset")
os.makedirs(os.path.join(dataset_dir, "i_s"), exist_ok=True)

base = Image.open(inp).convert("RGBA")
is_txt = os.path.join(dataset_dir, "i_s.txt")
it_txt = os.path.join(dataset_dir, "i_t.txt")

with open(is_txt, "w", encoding="utf-8") as fs, open(it_txt, "w", encoding="utf-8") as ft:
    for i, a in enumerate(areas, start=start):
        x1, y1, x2, y2 = map(int, a["bbox"])
        fname = f"{i:0{digits}d}.png"
        base.crop((x1, y1, x2, y2)).save(os.path.join(dataset_dir, "i_s", fname))
        fs.write(f"{fname} {a.get('source_text','')}\n")  # i_s.txt = 원문
        ft.write(f"{fname} {a.get('target_text','')}\n")  # i_t.txt = 번역

# tospi-dict('first : all in word convert to 'ㄱ')
# with open(is_txt, "w", encoding="utf-8") as fs, open(it_txt, "w", encoding="utf-8") as ft:
#     for i, a in enumerate(areas, start=start):
#         x1, y1, x2, y2 = map(int, a["bbox"])
#         fname = f"{i:0{digits}d}.png"
#         base.crop((x1, y1, x2, y2)).save(os.path.join(dataset_dir, "i_s", fname))

#         src = a.get("source_text", "")
#         tgt = a.get("target_text", "")
#         # 1) 단순히 길이만큼 'ㄱ'
#         masked = "ㄱ" * len(src)
#         # (선택) 공백은 보존하고 나머지만 'ㄱ'로 가리고 싶다면 아래로 교체:
#         # masked = "".join("ㄱ" if ch.strip() else ch for ch in tgt)

#         fs.write(f"{fname} {masked}\n")  # i_s.txt = 마스킹된 원문
#         ft.write(f"{fname} {tgt}\n")     # i_t.txt = 타겟 원문

print(dataset_dir, end="")
PY
)"

# 2) 추론: inference_sh.py 실행 (result는 out_root/<job_id>/result)
OUTPUT_DIR="$(dirname "$DATASET_DIR")/result"
mkdir -p "$OUTPUT_DIR"

"$PYTHON" /home/undergrad/model_base/TextCtrl-Translate/inference_sh.py \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$OUTPUT_DIR"

echo "OK: dataset=$DATASET_DIR  result=$OUTPUT_DIR"

# 3) 모든 영역 결과를 원본에 합성해서 all_result/composed.png 생성
ALL_RESULT_DIR="$(dirname "$DATASET_DIR")/all_result"
mkdir -p "$ALL_RESULT_DIR"

"$PYTHON" - "$JOB_JSON" "$DATASET_DIR" "$OUTPUT_DIR" "$ALL_RESULT_DIR" <<'PY'
import sys, os, json
from PIL import Image

job_json, dataset_dir, output_dir, all_result_dir = sys.argv[1:5]

with open(job_json, "r", encoding="utf-8") as f:
    job = json.load(f)

base = Image.open(job["input_path"]).convert("RGBA")
areas = job.get("areas", [])
digits = int(job.get("naming", {}).get("digits", 5))
start  = int(job.get("naming", {}).get("start", 0))

# 패치가 저장된 디렉토리(너 프로젝트에 맞게 바꿔도 됨)
patches_dir = output_dir

# 합성: 각 bbox 위치에 대응하는 패치 이미지를 붙인다
for idx, a in enumerate(areas, start=start):
    x1, y1, x2, y2 = map(int, a["bbox"])
    w, h = x2 - x1, y2 - y1
    fname = f"{idx:0{digits}d}.png"

    # 생성된 패치 경로 (변경 시 여기만 고치면 됨)
    patch_path = os.path.join(patches_dir, fname)

    if not os.path.exists(patch_path):
        # 패치가 없으면 스킵(원하면 raise로 바꿔도 됨)
        continue

    patch = Image.open(patch_path).convert("RGBA")
    if patch.size != (w, h):
        patch = patch.resize((w, h), Image.LANCZOS)

    # 알파가 있으면 마스크로 합성, 없으면 그냥 붙이기
    try:
        base.alpha_composite(patch, (x1, y1))
    except Exception:
        base.paste(patch, (x1, y1), patch if patch.mode == "RGBA" else None)

out_path = os.path.join(all_result_dir, "composed.png")
tmp = out_path + ".part.png"
base.save(tmp)
os.replace(tmp, out_path)
print(out_path, end="")
PY
