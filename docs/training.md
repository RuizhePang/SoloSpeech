# Training

The main training stages are in `run.sh`.

```bash
./run.sh --stage 2 --stop-stage 2  # compressor / VAE
./run.sh --stage 3 --stop-stage 3  # encode Libri2Mix with compressor
./run.sh --stage 4 --stop-stage 4  # extractor
./run.sh --stage 5 --stop-stage 5  # generate corrector data
./run.sh --stage 6 --stop-stage 6  # corrector
```

The default experiment config is `configs/tse/SoloSpeech.yaml`. Use `--config pretrained` or another Hydra config name to switch the full pipeline config.
