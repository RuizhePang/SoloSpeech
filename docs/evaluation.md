# Evaluation

Stage 7 evaluates compressor, extractor, corrector, and full-system metrics.

```bash
./run.sh --stage 7 --stop-stage 7
```

Evaluation settings live in `configs/tse/evaluation/default.yaml`. To override selected metrics:

```bash
./run.sh --stage 7 --stop-stage 7 --metrics system_sisdr,system_wer
```

For arbitrary test sets, provide a raw-audio manifest:

```bash
./run.sh --stage 7 --stop-stage 7 \
  --test_dir /path/to/testset \
  --test_manifest /path/to/manifest.csv
```

A manifest row should provide `mix`, `source`, and `reference` or their indexed forms such as `mix,source1,source2,reference1,reference2`. `transcript` fields are optional for WER.
