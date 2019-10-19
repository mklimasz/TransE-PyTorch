# TransE-PyTroch
Reimplementation of TransE model in PyTorch.

## Table of Contents
1. [Results](#results)
    1. [Metrics](#metrics)
    2. [Examples](#examples)
2. [Usage](#usage)
    1. [Training](#training)
        1. [Options](#options)
    2. [Unit tests](#unit-tests)
3. [References](#references)

## Results
Early stopping - the best Hit@10 metric on validation dataset.

### Metrics
All reported metrics are for **raw** test dataset.

### Examples

## Usage

### Training
```bash
python3 main.py --nouse_gpu
```
#### Options
To see possible configuration options run help
```bash
python3 main.py --help
```
### Unit tests
```bash
python3 -m unittest discover -p "*_test.py"
```

## References
```
@incollection{NIPS2013_5071,
    title = {Translating Embeddings for Modeling Multi-relational Data},
    author = {Bordes, Antoine and Usunier, Nicolas and Garcia-Duran, Alberto and Weston, Jason and Yakhnenko, Oksana},
    booktitle = {Advances in Neural Information Processing Systems 26},
    editor = {C. J. C. Burges and L. Bottou and M. Welling and Z. Ghahramani and K. Q. Weinberger},
    pages = {2787--2795},
    year = {2013},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf}
}
```