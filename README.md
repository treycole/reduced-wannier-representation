# Reduced Wannier Representation for Topological Bands [(arXiv:2412.17084)](https://arxiv.org/abs/2412.17084)

Code for the paper "Reduced Wannier Representation for Topological Bands" by Trey Cole and David Vanderbilt. 

## 📜 Paper
This paper has been published paper is avail
The preprint is available on [arXiv:2412.17084](https://arxiv.org/abs/2412.17084).

The `LaTeX` source code and a pdf of the arXiv manuscript can be found in the [paper](/paper) folder. 

## 💻 Results 

The dataset for the figures used in the paper is hosted on Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15186174.svg)](https://doi.org/10.5281/zenodo.15186174)

Results are generated using the notebooks and scripts found in the [calculations](/calculations) folder. The .py scripts are used for enhanced performance on more lengthy calculations, and the resulting figures are generated with the notebooks of the same name. The following packages are dependencies for running the code:

- `pythtb`
- `wanpy`

The `wanpy` package adds functionality for computing maximally localized Wannier functions in `pythtb`, but has not yet been included in a release. The version of `wanpy` used for this paper is linked as a submodule in this repository's root directory, or the most recent version can be found on the `wanpy` repository's [main page](https://github.com/treycole/WanPy). 

## 📚 Citation

If you use this code in your research, please cite our paper

**"Reduced Wannier Representation for Topological Bands"**  
*Trey Cole, David Vanderbilt*  
Phys. Rev. B 111, 205139, (2025)  
[DOI: 10.1103/PhysRevB.111.205139](https://doi.org/10.1103/PhysRevB.111.205139)

The `bibtex` for citing our paper can be copied here
```
@article{PhysRevB.111.205139,
  title = {Reduced Wannier representation for topological bands},
  author = {Cole, Trey and Vanderbilt, David},
  journal = {Phys. Rev. B},
  volume = {111},
  issue = {20},
  pages = {205139},
  numpages = {13},
  year = {2025},
  month = {May},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevB.111.205139},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.111.205139}
}

```

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
