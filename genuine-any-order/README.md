<div align="center">

# From Interface to Inference

### Eliciting Any-Order Inference from Any-Order Models

<br>

[Seunggeun Kim](https://seunggeunkimkr.github.io)<sup>1,\*</sup> &nbsp;&middot;&nbsp;
[Jaeyeon Kim](https://jaeyeonkim01.github.io)<sup>2,\*</sup> &nbsp;&middot;&nbsp;
[Taekyun Lee](https://taekyunl.github.io)<sup>1,\*</sup> &nbsp;&middot;&nbsp;
[Yuyuan Chen](https://yuyuanchen0.github.io)<sup>2,\*</sup>

Yilun Du<sup>2</sup> &nbsp;&middot;&nbsp;
Sham Kakade<sup>2</sup> &nbsp;&middot;&nbsp;
Sitan Chen<sup>2</sup>

<sup>1</sup> The University of Texas at Austin  <sup>2</sup> Harvard University  <sup>\*</sup> Co-first authors

<br>

[![arXiv](https://img.shields.io/badge/arXiv-2605.12466-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.26504)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-22C55E?style=for-the-badge)](LICENSE)

<br>

</div>

</div>

> MDMs promise any-order generation but often collapse to left-to-right decoding. We introduce **LatentMDM** and **FlexMDM** that enable genuinely any-order inference.

## Repository structure

| Directory | Contents |
|---|---|
| [`FlexMDM/`](FlexMDM/) | FlexMDM fine-tuned from Dream-Coder-7B-Base for any-order Python code generation: training, data pipeline, inference, and the full evaluation suite (pass@k + any-order metrics). Self-contained, with its own docs and environment. |
| [`LatentMDM/`](LatentMDM/) | LatentMDM training, sampling, and evaluation on TinyGSM. |

**Released checkpoint:** the FlexMDM model is on the Hugging Face Hub at
[`yuyuanchen0/flexmdm`](https://huggingface.co/yuyuanchen0/flexmdm)
(inference weights; see [`FlexMDM/README.md`](FlexMDM/README.md) for loading
and for reproducing the paper's tables). The LatentMDM model will be released soon.

The repository is licensed under Apache-2.0 (see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE)); `FlexMDM/` additionally carries its own
[`NOTICE`](FlexMDM/NOTICE) detailing its derivation from Dream-Coder and the
training-data licenses (one dataset is non-commercial).

## Citation

```bibtex
@article{kim2026interface,
  title   = {From Interface to Inference: Eliciting Any-Order Inference from Any-Order Models},
  author  = {Kim, Seunggeun and Kim, Jaeyeon and Lee, Taekyun and Chen, Yuyuan
             and Du, Yilun and Kakade, Sham and Chen, Sitan},
  journal = {arXiv preprint arXiv:2607.26504},
  year    = {2026},
  eprint  = {2607.26504},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url     = {https://arxiv.org/abs/2607.26504}
}
```

## Acknowledgments

This codebase builds on [PUMA](https://github.com/JaeyeonKim01/PUMA) and [FlexMDM](https://github.com/brianlck/FlexMDM).
