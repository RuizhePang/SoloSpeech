# Quick Use

This repository is driven by `run.sh`. Set the stage range and optional overrides on the command line.

```bash
./run.sh --stage 0 --stop-stage 1
```

For a small Libri2Mix download during local testing, adjust:

```bash
./run.sh --stage 0 --stop-stage 0 --librimix_sample_ratio 1/1000
```

Evaluation can run from a manifest of raw audio without precomputed compressor or extractor outputs:

```bash
./run.sh --stage 7 --stop-stage 7 \
  --test_dir /path/to/testset \
  --test_manifest /path/to/manifest.csv
```
