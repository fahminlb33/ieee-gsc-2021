# Challenge 1 - Application Failure Prediction

## Requirements

I use Anaconda to create a virtual environment to run the scripts. You'll need these libraries in your environment:

- jupyter
- argparse
- numpy
- pandas
- matplotlib
- seaborn
- imbalanced-learn
- scikit-learn
- scikit-learn-intelex

## How to run

This directory contains three important files,

- `model_full.ipynb`, used to perform preliminary model training, also contains code to perform model evaluation.
- `train_test_grid_search.py`, used to perform grid search on random forest classifier and perform predictions on the unlabeled data.

The Jupyter notebook files (ipynb) can be run directly without any modifications (make sure the path to the files stays the same).
To run the `train_test_grid_search.py`, assuming you cloned this script without changing the directory structure,

```bash
> cwd
/home/ieee-gsc-2021/challenge1

> python train_test_grid_search.py dataset/train_data.csv dataset/test_data_unlabeled.csv submission1.csv --mode complete
... long output ...

> python train_test_grid_search.py dataset/train_data.csv dataset/test_data_unlabeled.csv submission2.csv --mode abridged
... long output ...

```

The predictions will be saved to the `submission.csv` files. You can choose which model you want to run by specifying `--mode` argument.
  