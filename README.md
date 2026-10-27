# Generative and Geometric Deep Learning
Homework implementations and writeups by Michael Incorvaia for ECE 6257, Georgia Tech, Fall 2026.

| Assignment | Topics | Materials |
|---|---|---|
| [HW1](homework/hw01) | Data generation, pixel independence, graphical models, NADE and MADE | Python code, LaTeX/PDF writeup, figures |
| [HW2](homework/hw02) | Bernoulli mixtures and EM, NICE, VAE | Python code, LaTeX/PDF writeup, sample figures |
| [HW3](homework/hw03) | GANs and energy-based models | Notebooks, historical environment, LaTeX/PDF writeup |

## Setup
Use a dedicated Python environment, then run `python -m pip install -r requirements.txt`. Dependencies are unpinned starting points, not a validated environment lock. See each assignment README for working directories and additional setup. HW3 has its own historical environment and compatibility constraints.

Training scripts use [Eugen Hotaj's pytorch-generative](https://github.com/EugenHotaj/pytorch-generative). That external checkout is intentionally not vendored here. No pretrained weights, downloaded MNIST archives, or training logs are included. Full training has not been rerun during organization.

## Related work
- [GDL benchmarks](https://github.com/IncorvaiaM1/gdl-benchmarks): task prompts, reference implementations, grading tests, and archived agent evidence.
- [Options pricing through generative market simulation](https://github.com/IncorvaiaM1/Options-Pricing-Through-Generative-Market-Simulation): separate final project.

## Scope and attribution
Original textbook PDFs, lecture slides, assignment handouts, duplicate submission archives, and team presentation decks are excluded. Notebook outputs were cleared; existing PDFs and figures preserve reported results. Notebook scaffolding may include course or external material; see [THIRD_PARTY.md](THIRD_PARTY.md) and [RIGHTS.md](RIGHTS.md).
