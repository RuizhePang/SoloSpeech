from pathlib import Path
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="OpenSound/SoloSpeech-models",
    local_dir="../pretrained",
)

src_dir = Path(local_dir).resolve()
dst_root = Path("../experiments/pretrained/ckpts").resolve()

dst_root.mkdir(parents=True, exist_ok=True)

for src in src_dir.glob("*"):
    if not src.is_file():
        continue

    name = src.name.lower()

    if "compressor" in name:
        dst_dir = dst_root / "compressor"
    elif "extractor" in name:
        dst_dir = dst_root / "extractor"
    elif "corrector" in name:
        dst_dir = dst_root / "corrector"
    else:
        # 不属于这三类的文件直接跳过
        continue

    dst_dir.mkdir(parents=True, exist_ok=True)

    dst = dst_dir / src.name

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    dst.symlink_to(src)
