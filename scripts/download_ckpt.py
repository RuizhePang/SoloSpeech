from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="OpenSound/SoloSpeech-models",
    local_dir="./pretrained_models/original"
)
