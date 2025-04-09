# Reduced Wannier Representation for Topological Bands [(arXiv:2412.17084)](https://arxiv.org/abs/2412.17084)

Code for the paper "Reduced Wannier Representation for Topological Bands" by Trey Cole and David Vanderbilt. 

## 📜 Paper
The preprint is available on [arXiv](https://arxiv.org/abs/2412.17084).

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
arXiv:2412.17084, (2024)  
[DOI: 10.48550/arXiv.2412.17084](https://doi.org/10.48550/arXiv.2412.17084)

You can also [download the citation file here](./CITATION.cff) or get the `bibtex` from 'Cite this repository' under 'About'.

The `bibtex` for citing our paper can also be copied here
```
@misc{cole2025reducedwannierrepresentationtopological,
      title={Reduced Wannier Representation for Topological Bands}, 
      author={Trey Cole and David Vanderbilt},
      year={2025},
      eprint={2412.17084},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mes-hall},
      url={https://arxiv.org/abs/2412.17084}, 
}
```

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
