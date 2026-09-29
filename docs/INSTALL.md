# Installation

## 1. Clone the repositories

```bash
git clone https://github.com/ml-stat-Sustech/verbalized-confidence-training.git verbalized-confidence-training
cd verbalized-confidence-training

git clone https://github.com/verl-project/verl.git verl-main
git -C verl-main checkout --detach 5b2bfe7c5860ed0f8d6fbd30eaf8f1c2f970f9cc
patch -d verl-main -p1 < patches/verl-save-hf-model-bf16.patch
```

## 2. Create the environment

```bash
conda create -n vc_verl python=3.12 -y
conda activate vc_verl

python -m pip install -U pip wheel
python -m pip install -r requirements.txt
# Prebuilt FlashAttention 2.8.3 for torch 2.11, CUDA 13, and Python 3.12.
# Alternatively, build it from source: python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install "https://github.com/adithyaxx/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
python -m pip install -e ./verl-main
```

See [RUN.md](./RUN.md) for data generation, SFT, RL, and evaluation commands.
