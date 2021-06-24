# Reproducible Deep Learning
## Extra: Auto-PyTorch for neural architecture search.
### Author: [AutoML Groups Freiburg and Hannover](http://www.automl.org/)
[[Official Auto-PyTorch Repository](https://github.com/automl/Auto-PyTorch/)]


## Installation

Clone repository

```sh
$ cd install/path
$ git clone https://github.com/automl/Auto-PyTorch.git
$ cd Auto-PyTorch
```
If you want to contribute to this repository switch to our current development branch

```sh
$ git checkout development
```

Install pytorch: 
https://pytorch.org/

Install Auto-PyTorch:

```sh
$ cat requirements.txt | xargs -n 1 -L 1 pip install
$ python setup.py install
```




## License

This program is free software: you can redistribute it and/or modify
it under the terms of the Apache license 2.0 (please see the LICENSE file).

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

You should have received a copy of the Apache license 2.0
along with this program (see LICENSE file).

## Reference

```bibtex
  @article{zimmer-tpami21a,
  author = {Lucas Zimmer and Marius Lindauer and Frank Hutter},
  title = {Auto-PyTorch Tabular: Multi-Fidelity MetaLearning for Efficient and Robust AutoDL},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year = {2021},
  note = {IEEE early access; also available under https://arxiv.org/abs/2006.13799},
  pages = {1-12}
}
```

```bibtex
@incollection{mendoza-automlbook18a,
  author    = {Hector Mendoza and Aaron Klein and Matthias Feurer and Jost Tobias Springenberg and Matthias Urban and Michael Burkart and Max Dippel and Marius Lindauer and Frank Hutter},
  title     = {Towards Automatically-Tuned Deep Neural Networks},
  year      = {2018},
  month     = dec,
  editor    = {Hutter, Frank and Kotthoff, Lars and Vanschoren, Joaquin},
  booktitle = {AutoML: Methods, Sytems, Challenges},
  publisher = {Springer},
  chapter   = {7},
  pages     = {141--156}
}
```

## Contact

Auto-PyTorch is developed by the [AutoML Groups of the University of Freiburg and Hannover](http://www.automl.org/).