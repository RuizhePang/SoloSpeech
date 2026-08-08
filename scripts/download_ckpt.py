from pathlib import Path
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="OpenSound/SoloSpeech-models",
    local_dir="../pretrained"
)

scr_dir = Path(local_dir).resolve()
dst_dir = Path("../experiments/pretrained/ckpts").resolve()

dst_dir.mkdir(parents=True, exist_ok=True)

for src in src_dir.glob("*"):
    if not src.is_file():
        continue
    relative_path = src.relative_to(src_dir)
    dst = dst_dir / relative_path

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    dst.symlink_to(src)
