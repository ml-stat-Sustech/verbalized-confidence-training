<h1 align="center">On the Pitfalls of Verbalized Confidence Priors for Calibrating Large Reasoning Models</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2609.32470"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg?logo=arxiv" alt="Paper"></a>
  <a href="https://huggingface.co/collections/SUSTech/calibsft-6abb0bd95e722b92b235c452"><img src="https://img.shields.io/badge/Models-Hugging%20Face-ffd21e.svg?logo=huggingface" alt="Models"></a>
  <a href="https://huggingface.co/datasets/SUSTech/CalibSFT-DeepScaleR"><img src="https://img.shields.io/badge/Dataset-Hugging%20Face-ffd21e.svg?logo=huggingface" alt="Dataset"></a>
</p>

This is the official implementation of [CalibSFT](https://arxiv.org/abs/2609.32470), a plug-and-play supervised fine-tuning stage that shapes a calibrated, broadly supported verbalized-confidence prior before confidence-aware reinforcement learning. For each training question, CalibSFT samples $n$ responses from the base model and labels each response with the confidence $c^* = \lambda q + (1-\lambda) z$, where $q$ is the question's success rate and $z$ is the response's correctness. It then balances the data across confidence levels from 0 to 1 and supervises confidence on all responses, but reasoning and answers only on correct ones. The resulting checkpoint initializes confidence-aware RL methods such as RLCR, CoCA, DCPO, and ReDoubt.

## Installation

Please refer to [docs/INSTALL.md](./docs/INSTALL.md) to set up the environment.

## Quick Start

We release the [CalibSFT](https://huggingface.co/SUSTech/Qwen3-8B-CalibSFT) and [CalibSFT → RLCR](https://huggingface.co/SUSTech/Qwen3-8B-CalibSFT-RLCR) checkpoints for quick reproduction. Evaluate them on all 16 benchmarks (temperature 0.6; 4 responses per question, 32 on AIME):

**Qwen3-8B**

```bash
CKPT=Qwen/Qwen3-8B \
bash scripts/examples/eval/base.sh
```

**CalibSFT**

```bash
CKPT=SUSTech/Qwen3-8B-CalibSFT \
bash scripts/examples/eval/calib_sft.sh
```

**CalibSFT → RLCR**

```bash
CKPT=SUSTech/Qwen3-8B-CalibSFT-RLCR \
bash scripts/examples/eval/calib_sft_rlcr.sh
```

## Training

Please refer to [docs/RUN.md](./docs/RUN.md) for CalibSFT data generation, CalibSFT training, confidence-aware RL, and evaluation of trained checkpoints.

## Citation

```bibtex
@article{wang2026pitfalls,
  title={On the Pitfalls of Verbalized Confidence Priors for Calibrating Large Reasoning Models},
  author={Wang, Shuoyuan and Luo, Beier and Zeng, Hao and Yu, Chengyao and Zhang, Songxin and Xie, Zejian and Jing, Bingyi and Wei, Hongxin},
  journal={arXiv preprint arXiv:2609.32470},
  year={2026}
}
```
