# Black Box FDR

This is a reference implementation for Black Box FDR (BB-FDR). BB-FDR is an empirical-Bayes method for analyzing multi-experiment studies when many covariates are gathered per experiment. It performs two stages of selection:

- *Stage 1*: Fit a (black box) neural network model using the covariates for each experiment. The NN is then used to select the significant outcomes at a given false discovery rate (FDR).

- *Stage 2*: For each covariate, fit a (black box) gradient boosting classifier to predict the probability of a covariate given other covariates. The classifier is then used to perform a conditional randomization test (CRT) to determine whether that mutation is significantly associated with differential response.

Note that the code is setup to be run easily on a cluster, in case one has hundreds of such studies to analyze. The model is checkpointed frequently in order to allow for preemption without losing the progress.

## Requirements

```
numpy
scipy
pytorch
sklearn
```

We use pytorch to fit the neural network prior model and scikit-learn to fit our gradient boosting trees.

## Case study

We provide an example of using BB-FDR to analyze dose-response data from the Genomics of Drug Sensitivity in Cancer. See `python/cancer.py` for details.

## Citing BB-FDR

If you use this code, please cite:

```
@inproceedings{tansey:etal:icml:2018:bbfdr,
  title={Black Box {{FDR}}},
  author={Tansey, W. and Wang, Y. and Blei, D. B. and Rabadan, R.},
  booktitle={International Conference on Machine Learning (ICML'18)},
  year={2018}
}
```
