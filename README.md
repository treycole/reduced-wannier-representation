# Reduced Wannier Representation for Topological Bands [(arXiv:2412.17084)](https://arxiv.org/abs/2412.17084)
Code for the paper "Reduced Wannier Representation for Topological Bands" by Trey Cole and David Vanderbilt. 

The `LaTeX` source code and a pdf of the arXiv manuscript can be found in the [paper](/paper) folder. Results are generated using the notebooks and scripts found in the [calculations](/calculations) folder. The .py scripts are used for enhanced performance on more lengthy calculations, and the resulting figures are generated with the notebooks of the same name. The following packages are dependencies for running the code:
- `pythtb`
- `wanpy`

The `wanpy` package adds functionality for computing maximally localized Wannier functions in `pythtb`, but has not yet been included in a release. The repository for `wanpy` is linked in this directory or can be found [here](https://github.com/treycole/WanPy). 

The dataset for the figures used in the paper is hosted on Zenodo with DOI 10.5281/zenodo.14544685.


Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
