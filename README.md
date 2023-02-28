# Improving the Performance of Robust Control through Event-Triggered Learning

This repository is part of the supplementary material for the submission titled **Improving the Performance of Robust Control through Event-Triggered Learning** by *Alexander von Rohr, Friedrich Solowjow and Sebastian Trimpe* published in the [proceedings of the IEEE Conference on Decision and Control](https://ieeexplore.ieee.org/document/9993350).

If you are finding this code useful please get in contact and consider citing the paper.

```
@inproceedings{rohr2022improving,
  author = {{von Rohr}, Alexander and Solowjow, Friedrich and Trimpe, Sebastian},
  title = {Improving the Performance of Robust Control through Event-Triggered Learning},
  booktitle = {Proceedings of the IEEE Conference on Decision and Control},
  pages = {3424-3430},
  year = {2022},
  doi = {10.1109/CDC51059.2022.9993350},
}
```


## How to use the supplementary code

### Install dependencies

This project uses pipenv (https://pypi.org/project/pipenv/) to manage dependencies
I recommend using pyenv (https://github.com/pyenv/pyenv) to manage your python version.

When you have pipenv and the correct python version installed run

```
pipenv install
```

You als need to have MOSEK (https://www.mosek.com/) installed for the LMI based synthesis.
We use PICOS (https://pypi.org/project/PICOS/) as interface to the underlying solver. 
That means, in principle, it is possible to replace MOSEK with CVXOPT (https://cvxopt.org/) without many changes.

Once you have installed all dependencies you can start the python virtual environment:

```
pipenv shell
```

### Reproducing the figures

The data presented in the paper is part of this repository and can be found in the *data* folder.
To reproduce the figures presented in the paper you can re-run the scripts named *plot_\**.

### Reproducing results

The results of Section IV.A can be reproduced with the script *cost_optimal_excitation_1d.py*.

The results of Section IV.B can be reproduced with the script *improve_robust_control.py*.
