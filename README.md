# T-REX SELECTOR

**FDR-controlled variable selection in high-dimensional regression and classification.
A C++20 library with R and Python bindings.**

---

## A Short Story about the T-Rex Selector

False discoveries plague science and engineering.
In genomics, a single false discovery can wrongly tie a gene to a disease, contributing to the
reproducibility crisis and derailing years of research and clinical effort [(Huffman et al., 2018)](https://www.nature.com/articles/s41467-018-07348-x).
In finance, hundreds of studies claim to have found factors that explain stock returns, yet most of
these findings are likely false [(Harvey, Liu, and Zhu, 2016)](https://academic.oup.com/rfs/article-abstract/29/1/5/1843824?redirectedFrom=fulltext).

The T-Rex Selector performs high-dimensional variable selection while controlling the rate of false
discoveries at a level you choose, with rigorous finite-sample guarantees, and fast and
memory-efficient enough to run on millions of variables [(Machkour, Muma, and Palomar, 2025)](https://www.sciencedirect.com/science/article/pii/S016516842500009X).

The algorithm works by generating random dummy variables that act as a calibration baseline:
during each of *K* independent random experiments, a terminating solver traces solution paths
through both the real predictors and the dummies.
By comparing how often each real predictor is selected relative to the dummies across all
experiments, a voting threshold is derived that provably bounds the FDR.

**Project website:** [trexselector.com](https://trexselector.com)

---

## Implemented Variants

| Class                          | Description                                               |
|--------------------------------|-----------------------------------------------------------|
| `TRexSelector`                 | Core FDR-controlled selector for linear regression        |
| `TRexDASelector`               | Dependency-aware variant for correlated predictors        |
| `TRexGVSSelector`              | Grouped variable selection                                |
| `TRexScreeningSelector`        | Fast FDR-controlled variable pre-screening for large *p*  |
| `TRexBiobankScreeningSelector` | High-throughput genomic / biobank screening               |
| `TRexSPCA`                     | FDR-controlled sparse principal component analysis        |

---

## Repository Layout

This repository hosts the C++ core and R and Python language packages.
Results are identical across languages for the same seeds and parameters.
Each component has its own README with installation instructions, examples, and developer
documentation:

* **C++ core library** — [`cpp/`](cpp/), documented in [cpp/README.md](cpp/README.md):
  building the library and tests, and reaching it from your own C++ project via
  `find_package(TRexSelector)`.
* **Python package** — [`Python/TRexSelectorNeo/`](Python/TRexSelectorNeo/), documented in
  [Python/TRexSelectorNeo/README.md](Python/TRexSelectorNeo/README.md): installation, quick
  start, and development setup. Distributed as `TRexSelectorNeo`, imported as
  `trex_selector_neo`.
* **R package** — [`R/TRexSelectorNeo/`](R/TRexSelectorNeo/), documented in
  [R/TRexSelectorNeo/README.md](R/TRexSelectorNeo/README.md): installation, quick start, and
  vignette.

---

## References

* Machkour, Muma, Palomar - *The Terminating-Random Experiments Selector: Fast High-Dimensional Variable Selection with False Discovery Rate Control*, Signal Processing, vol. 231, pp. 109894, 2025. [DOI link](https://www.sciencedirect.com/science/article/pii/S016516842500009X)
* Machkour, Muma, Palomar - *High-dimensional false discovery rate control for dependent variables*, Signal Processing, vol. 234, pp. 109990, 2025. [DOI link](https://www.sciencedirect.com/science/article/pii/S0165168425001045)
* Taulant Koka, Jasin Machkour, Daniel P. Palomar, and Michael Muma, “Virtual Dummies: Enabling Scalable FDR-Controlled Variable Selection via Sequential Sampling of Null Features,” arXiv:2604.07464, 2026. [DOI link](https://arxiv.org/abs/2604.07464)
* Huffman, J. E., et al. - *Examining the current standards for genetic discovery and replication in the era of mega-biobanks*, Nature Communications, 9(1), 5054 (2018). [DOI link](https://www.nature.com/articles/s41467-018-07348-x)
* Harvey, C. R., Liu, Y., and Zhu, H. - *… and the Cross-Section of Expected Returns*, The Review of Financial Studies, 29(1), 5–68 (2016). [DOI link](https://academic.oup.com/rfs/article-abstract/29/1/5/1843824?redirectedFrom=fulltext)

---

## License

GPL ≥ 3 — see [LICENSE](LICENSE).
