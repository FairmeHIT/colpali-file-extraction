# Local ColPali + ViDoRe V1 Reproduction

This workspace contains:

- `colpali`: official `illuin-tech/colpali` checked out at `v0.1.1`
- `vidore-benchmark`: official `illuin-tech/vidore-benchmark` checked out at `v3.3.0`
- `colpali-repro`: referenced third-party reproduction repo
- `.venv-vidore`: Python 3.12 environment using local CUDA/PyTorch wheels from `/mnt/d/codes/torch_install_package`
- `env.sh` sets `HF_ENDPOINT=https://hf-mirror.com` by default for Hugging Face downloads.

Use:

```bash
source env.sh
vidore-benchmark --help
```

Smoke run on one ViDoRe V1 dataset with a smaller non-gated visual baseline:

```bash
bash run_vidore_v1_siglip_smoke.sh
```

Full ViDoRe V1 collection run with the same smaller baseline:

```bash
bash run_vidore_v1_siglip_full.sh
```

ColPali smoke run on one ViDoRe V1 dataset:

```bash
bash run_vidore_v1_smoke.sh
```

Full ColPali ViDoRe V1 collection run:

```bash
bash run_vidore_v1_full.sh
```

Outputs are written under `outputs/`. Hugging Face model and dataset cache is under `hf-cache/`.

ColPali V1 requires access to the gated `google/paligemma-3b-mix-448` model. Use the SigLIP scripts first if you want a smaller, non-gated first reproduction.
