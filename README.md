# Reduced Wannier representation for topological bands 

[![Journal](https://img.shields.io/badge/PhysRevB-10.1103/PhysRevB.111.205139-blue)](https://doi.org/10.1103/PhysRevB.111.205139) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15186174.svg)](https://doi.org/10.5281/zenodo.15186174)

Code for the paper "Reduced Wannier representation for topological bands" by Trey Cole and David Vanderbilt. 

## 📜 Paper
This paper has been published in Physical Review B and is available on the APS website [(Phys. Rev. B 111, 205139)](https://doi.org/10.1103/PhysRevB.111.205139).

The arXiv preprint is available at [arXiv:2412.17084](https://arxiv.org/abs/2412.17084).

The `LaTeX` source code and a PDF of each version of the arXiv manuscript can be found in the [paper](/paper) folder. 

## 💻 Results 

The dataset for the figures used in the paper is hosted on [Zenodo](https://doi.org/10.5281/zenodo.15186174). Download the data and place it in the [calculations](/calculations) to reproduce the results.

Results are generated using the notebooks and scripts located in the [calculations](/calculations) folder. The Python scripts are used to improve performance for longer calculations, while the corresponding Jupyter notebooks produce the final figures. 

The [wanpy](/wanpy) folder provides functionality for computing maximally localized Wannier functions in `pythtb`. 

All required Python dependencies are listed in [requirements.txt](requierements.txt). Install them with

```bash
pip install -r requirements.txt
```

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

## Licenses

[![Code: GPL v3](https://img.shields.io/badge/Code-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Paper: CC BY 4.0](https://img.shields.io/badge/Paper-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
