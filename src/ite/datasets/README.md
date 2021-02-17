# Datasets

This module contains code for loading and pre-processing the datasets.

## API
1. `load` - returns sampled train and test sets from the target dataset.

Args:

 - `dataset`: The dataset to sample from. Example: "twins".
 - `train_split`: Float number from [0, 1] indicating the train-test split ratio. Defaults to 0.8.
 - `downsample`: Optional integer indicating an upper bound for the set size. For example, for CMGP, we downsample to 1000 elements.


## Supported datasets
1. [__Twins__](https://www.nber.org/data/linked-birth-infant-death-data-vital-statistics-data.html). Implementation adapted from the [original GANITE dataset loader](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/alg/ganite/data_preprocessing_ganite.py).