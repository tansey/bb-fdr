'''Conditional randomization test (CRT) implementation for whether a variable
has any effect on the posterior probability of coming from the alternative.'''
from __future__ import print_function
import sys
import os
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from utils import fit_logistic, bh_predictions

class PosteriorConditionalRandomizationTester:
    def __init__(self, prior, fdr=0.1):
        self.prior = prior
        self.fdr = fdr

        # Estimate the p-values for each feature
        self.p_values = np.zeros(self.prior.nfeatures)
        self.tstats = np.zeros(self.prior.nfeatures)
        self.null_probs = np.zeros((self.prior.nsamples, self.prior.nfeatures))

        self.p_values[:] = np.nan
        self.null_probs[:] = np.nan

    def run(self, feat_idx, trials_per_feature=200,
                    verbose=False, working_path=None,
                    model='gboost', t_chunks=20, p_cutoff=0.1,
                    feature_name=None, save_freq=1,
                    entropy_tol=0):
        tstats_null = np.zeros(trials_per_feature)

        # Predict the conditional probabilities of each feature
        if working_path is not None and os.path.exists(working_path + '_tnull{}.npy'.format(feat_idx)):
            if verbose:
                print('Found existing null t-stats. Loading...')
            try:
                tstats_finished = np.load(working_path + '_tnull{}.npy'.format(feat_idx))
                tstats_null[:tstats_finished.shape[0]] = tstats_finished
                start_trial = tstats_finished.shape[0]
            except:
                print('Corrupted. Starting over...')
                start_trial = 0
        else:
            start_trial = 0

        if feature_name is None:
            feature_name = str(feat_idx+1)

        # Train and predict using holdout sets
        if verbose:
            print('Fitting {} conditional model for feature {}'.format(model, feature_name))
            sys.stdout.flush()
        for fold_idx, test_indices in enumerate(self.prior.folds):
            if verbose:
                print('\t\tFeature {} Fold {}'.format(feature_name, fold_idx+1))
                sys.stdout.flush()
            # Create train/validate splits
            imask = np.ones(self.prior.nsamples, dtype=bool)
            imask[test_indices] = False

            jmask = np.ones(self.prior.nfeatures, dtype=bool)
            jmask[feat_idx] = False
            
            if verbose:
                print('\t\t\tFitting conditional model')
            # Fit a linear model
            self.null_probs[test_indices, feat_idx], _, _ = fit_logistic(self.prior.X[imask][:,jmask],
                                                     self.prior.X[imask, feat_idx],
                                                     X_holdout=self.prior.X[test_indices][:,jmask],
                                                     model=model)

        if verbose:
            print('\t\t\tDrawing max of {} null samples'.format(trials_per_feature))

        # Sample and recalculate the posterior each time
        self.tstats[feat_idx] = (self.prior.posteriors*np.log(self.prior.posteriors) + (1-self.prior.posteriors)*np.log(1-self.prior.posteriors)).mean()
        X_null = np.copy(self.prior.X)
        null_posteriors = np.zeros_like(self.prior.posteriors)
        for trial in range(start_trial, trials_per_feature):
            for fold_idx, test_indices in enumerate(self.prior.folds):
                X_null[test_indices,feat_idx] = (np.random.random(size=len(test_indices)) <= self.null_probs[test_indices][:, feat_idx]).astype(int)

                # Use the prior to calculate the null posteriors
                _, null_posteriors[test_indices] = self.prior.predict(X_null[test_indices], y=self.prior.y[test_indices], models=[fold_idx])

            # Update the t-statistic (negative marginal entropy)
            tstats_null[trial] = (null_posteriors*np.log(null_posteriors) + (1-null_posteriors)*np.log(1-null_posteriors)).mean()

            if (trial+1) % t_chunks == 0:
                if verbose:
                    print('\t\t\tNull {}'.format(trial+1))
                self.p_values[feat_idx] = ((tstats_null[:trial+1]+entropy_tol) >= self.tstats[feat_idx]).mean()
                if self.p_values[feat_idx] > p_cutoff:
                    if verbose:
                        print('\t\t\tFeature {} p-value over threshold (p={}). Stopping early.'.format(feature_name, self.p_values[feat_idx]))
                        print('True t-stat: {}'.format(self.tstats[feat_idx]))
                        print(tstats_null[:trial+1])
                    return
            
            if working_path is not None and (trial % save_freq) == 0:
                np.save(working_path + '_tnull{}'.format(feat_idx), tstats_null[:trial+1])

        self.p_values[feat_idx] = (tstats_null >= self.tstats[feat_idx]).mean()
        if verbose:
            print('Feature {} t-stat: {} p-value: {}'.format(feature_name, self.tstats[feat_idx], self.p_values[feat_idx]))
            print(tstats_null)
            sys.stdout.flush()

    def predictions(self):
        self.discoveries = bh_predictions(self.p_values, self.fdr)
        return {'tstats': self.tstats,
                'p_values': self.p_values,
                'discoveries': self.discoveries}












