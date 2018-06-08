'''Black box false discovery rate (FDR) control for treatment effects. We use
the two-groups empirical Bayes approach for the treatment effects and fit a
collection of deep networks as the empirical prior.'''
from __future__ import print_function
import sys
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm, beta
from utils import p2z, tpr, fdr, true_positives, false_positives, create_folds,\
                  batches, calc_fdr, p_value_2sided
from normix import predictive_recursion, empirical_null, GridDistribution

class LinearAdaptiveFDRModeler(nn.Module):
    def __init__(self, nfeatures):
        super(LinearAdaptiveFDRModeler, self).__init__()
        self.fc_in = nn.Sequential(nn.Linear(nfeatures, 2), nn.Softplus())
    
    def forward(self, x):
        return self.fc_in(x) + 1.


class DeepAdaptiveFDRModeler(nn.Module):
    def __init__(self, nfeatures):
        super(DeepAdaptiveFDRModeler, self).__init__()
        self.fc_in = nn.Sequential(
                nn.Linear(nfeatures, 200),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(200, 200),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(200, 2),
                nn.Softplus())
    
    def forward(self, x):
        return self.fc_in(x) + 1.

class BlackBoxTwoGroupsModel:
    def __init__(self, X, y, fdr=0.1, num_alt_bins=220,
                       num_pr_sweeps=5, estimate_null=False,
                       pvalues=False, beta_gridsize=1001):
        if pvalues:
            # Convert the p-values to z-scores
            print('\tConverting to z-scores under a one-sided test assumption')
            sys.stdout.flush()
            p_values = np.copy(y)
            y = p2z(p_values)
        else:
            p_values = p_value_2sided(y)

        self.X = X
        self.y = y
        self.nsamples = X.shape[0]
        self.nfeatures = X.shape[1]
        self.fdr = fdr

        self.X_means = X.mean(axis=0)[np.newaxis,:]
        self.X_std = X.std(axis=0)[np.newaxis,:]

        # Empirical null estimation
        if estimate_null:
            print('\tEstimating empirical null distribution')
            sys.stdout.flush()
            mu0, sigma0 = empirical_null(y)
            p_values = p_value_2sided(y, mu0, sigma0)
        else:
            mu0, sigma0 = 0., 1.
        self.null_dist = (mu0, sigma0)
        # print('\tNull mean: {} std: {}'.format(mu0, sigma0))
        # sys.stdout.flush()

        # Predictive recursion estimate of alternative distribution
        min_alt_z, max_alt_z = min(-10, y.min() - 1), max(y.max() + 1, 10)
        # print('\tEstimating alternative distribution via predictive recursion over range [{},{}] with {} bins'.format(min_alt_z, max_alt_z, num_alt_bins))
        sys.stdout.flush()
        grid_x = np.linspace(min_alt_z, max_alt_z, num_alt_bins)
        pr_results = predictive_recursion(y, num_pr_sweeps, grid_x, mu0=mu0, sig0=sigma0)
        self.pi0 = pr_results['pi0']
        self.alt_dist = GridDistribution(pr_results['grid_x'], pr_results['y_signal'])

        # Create a discrete grid approximation to the Beta support
        self.beta_grid = np.linspace(0.001, 0.999, beta_gridsize)[np.newaxis,:]

        # Cache the likelihoods
        # print('\tCaching likelihoods')
        sys.stdout.flush()
        self.P0 = norm.pdf(y, mu0, sigma0)[:,np.newaxis]
        self.P1 = self.alt_dist.pdf(y)[:,np.newaxis]

        # Create the torch variables
        self.tP0 = autograd.Variable(torch.FloatTensor(self.P0), requires_grad=False)
        self.tP1 = autograd.Variable(torch.FloatTensor(self.P1), requires_grad=False)
        self.tX = autograd.Variable(torch.FloatTensor((self.X - self.X_means) / self.X_std), requires_grad=False)


    def train(self, model_fn=None, lasso=0., l2=1e-4, lr=3e-4, num_epochs=250,
                    batch_size=None, num_folds=3, val_pct=0.1, verbose=False, folds=None,
                    weight_decay=0.01, random_restarts=1, save_dir='/tmp/',
                    momentum=0.9, patience=3, clip_gradients=None):
        # Make sure we have a model of the prior
        if model_fn is None:
            model_fn = lambda nfeatures: DeepAdaptiveFDRModeler(nfeatures)

        # Lasso penalty (if any)
        lasso = autograd.Variable(torch.FloatTensor([lasso]), requires_grad=False)
        l2 = autograd.Variable(torch.FloatTensor([l2]), requires_grad=False)

        if batch_size is None:
            batch_size = int(max(10,min(100,np.round(self.X.shape[0] / 100.))))
            print('Batch size: {}'.format(batch_size))

        # Discrete approximation of a beta PDF support
        tbeta_grid = autograd.Variable(torch.FloatTensor(self.beta_grid), requires_grad=False)
        sys.stdout.flush()
        # Split the data into a bunch of cross-validation folds
        if folds is None:
            if verbose:
                print('\tCreating {} folds'.format(num_folds))
                sys.stdout.flush()
            folds = create_folds(self.X, k=num_folds)
        self.priors = np.zeros((self.nsamples,2), dtype=float)
        self.models = []
        train_losses, val_losses = np.zeros((len(folds),random_restarts,num_epochs)), np.zeros((len(folds),random_restarts,num_epochs))
        epochs_per_fold = np.zeros(len(folds))
        for fold_idx, test_indices in enumerate(folds):
            # Create train/validate splits
            mask = np.ones(self.nsamples, dtype=bool)
            mask[test_indices] = False
            indices = np.arange(self.nsamples, dtype=int)[mask]
            np.random.shuffle(indices)
            train_cutoff = int(np.round(len(indices)*(1-val_pct)))
            train_indices = indices[:train_cutoff]
            validate_indices = indices[train_cutoff:]
            torch_test_indices = autograd.Variable(torch.LongTensor(test_indices), requires_grad=False)
            best_loss = None

            # Try re-initializing a few times
            for restart in range(random_restarts):
                model = model_fn(self.nfeatures)
                
                # Setup the optimizers
                # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
                # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience)
                optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
                # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                # Train the model
                for epoch in range(num_epochs):
                    if verbose:
                        print('\t\tRestart {} Fold {} Epoch {}'.format(restart+1, fold_idx+1,epoch+1))
                        sys.stdout.flush()

                    train_loss = torch.Tensor([0])
                    for batch_idx, batch in enumerate(batches(train_indices, batch_size, shuffle=False)):
                        if verbose and (batch_idx % 100 == 0):
                            print('\t\t\tBatch {}'.format(batch_idx))
                        tidx = autograd.Variable(torch.LongTensor(batch), requires_grad=False)

                        # Set the model to training mode
                        model.train()

                        # Reset the gradient
                        model.zero_grad()

                        # Run the model and get the prior predictions
                        concentrations = model(self.tX[tidx])

                        # Calculate the loss as the negative log-likelihood of the data
                        # Use a beta prior for the treatment effect
                        prior_dist = torch.distributions.Beta(concentrations[:,0:1], concentrations[:,1:2])

                        # Discretize the (0,1) interval to approximate the beta PDF
                        prior_probs = prior_dist.log_prob(tbeta_grid).exp()
                        prior_probs = prior_probs / prior_probs.sum(dim=1, keepdim=True)

                        # Calculate the loss
                        posterior_probs = (((1-tbeta_grid) * self.tP0[tidx]
                                             + tbeta_grid * self.tP1[tidx]) * prior_probs).sum(dim=1)
                        loss = -posterior_probs.log().mean()

                        # L1 penalty to shrink c and be more conservative
                        regularized_loss = loss + lasso * concentrations.mean() + l2 * (concentrations**2).mean()

                        # Update the model with gradient clipping for stability
                        regularized_loss.backward()

                        # Clip the gradients if need-be
                        if clip_gradients is not None:
                            torch.nn.utils.clip_grad_norm(model.parameters(), clip_gradients)

                        # Apply the update
                        [p for p in model.parameters() if p.requires_grad]
                        optimizer.step()

                        # Track the loss
                        train_loss += loss.data

                    validate_loss = torch.Tensor([0])
                    for batch_idx, batch in enumerate(batches(validate_indices, batch_size)):
                        if verbose and (batch_idx % 100 == 0):
                            print('\t\t\tValidation Batch {}'.format(batch_idx))
                        tidx = autograd.Variable(torch.LongTensor(batch), requires_grad=False)

                        # Set the model to test mode
                        model.eval()

                        # Reset the gradient
                        model.zero_grad()

                        # Run the model and get the prior predictions
                        concentrations = model(self.tX[tidx])

                        # Calculate the loss as the negative log-likelihood of the data
                        # Use a beta prior for the treatment effect
                        prior_dist = torch.distributions.Beta(concentrations[:,0:1], concentrations[:,1:2])

                        # Discretize the (0,1) interval to approximate the beta PDF
                        prior_probs = prior_dist.log_prob(tbeta_grid).exp()
                        prior_probs = (prior_probs / prior_probs.sum(dim=1, keepdim=True)).clamp(1e-8, 1-1e-8)

                        # Calculate the loss
                        posterior_probs = (((1-tbeta_grid) * self.tP0[tidx]
                                             + tbeta_grid * self.tP1[tidx]) * prior_probs).sum(dim=1).clamp(1e-8, 1-1e-8)
                        loss = -posterior_probs.log().sum()
                        
                        # Track the loss
                        validate_loss += loss.data

                    train_losses[fold_idx, restart, epoch] = train_loss.numpy() / float(len(train_indices))
                    val_losses[fold_idx, restart, epoch] = validate_loss.numpy() / float(len(validate_indices))

                    # # Adjust the learning rate down if the validation performance is bad
                    # scheduler.step(val_losses[fold_idx, epoch])

                    # Check if we are currently have the best held-out log-likelihood
                    if verbose:
                        print('Validation loss: {} Best: {}'.format(val_losses[fold_idx, restart, epoch], best_loss))
                    if (restart == 0 and epoch == 0) or val_losses[fold_idx, restart, epoch] <= best_loss:
                        if verbose:
                            print('\t\t\tSaving test set results.      <----- New high water mark for fold {} on epoch {}'.format(fold_idx+1, epoch+1))
                        # If so, use the current model on the test set
                        best_loss = val_losses[fold_idx, restart, epoch]
                        epochs_per_fold[fold_idx] = epoch + 1
                        self.priors[test_indices] = model(self.tX[torch_test_indices]).data.numpy()
                        torch.save(model, save_dir + '_fold{}.pt'.format(fold_idx))

                    if verbose:
                        means = self.priors[test_indices,0] / self.priors[test_indices].sum(axis=1)
                        print('Prior range: [{},{}]'.format(means.min(), means.max()))
                        print('First 3:')
                        print(self.priors[test_indices][:3])

            # Reload the best model
            self.models.append(torch.load(save_dir + '_fold{}.pt'.format(fold_idx)))
                
        # Calculate the posterior probabilities
        if verbose:
            print('Calculating posteriors.')
            sys.stdout.flush()
        prior_grid = beta.pdf(self.beta_grid, self.priors[:,0:1], self.priors[:,1:2])
        prior_grid /= prior_grid.sum(axis=1, keepdims=True)
        post0 = self.P0 * (1-self.beta_grid)
        post1 = self.P1 * self.beta_grid
        self.posteriors = ((post1 / (post0 + post1)) * prior_grid).sum(axis=1)
        self.posteriors = self.posteriors.clip(1e-8,1-1e-8)

        if verbose:
            print('Calculating predictions at a {:.2f}% FDR threshold'.format(self.fdr*100))
            sys.stdout.flush()
        self.predictions = calc_fdr(self.posteriors, self.fdr)

        if verbose:
            print('Finished training.')
            sys.stdout.flush()

        self.folds = folds

        return {'train_losses': train_losses,
                'validation_losses': val_losses,
                'priors': self.priors,
                'posteriors': self.posteriors,
                'predictions': self.predictions,
                'models': self.models,
                'folds': folds}

    def predict(self, X, y=None, models=None, batch_size=100):
        # Potentially use a subset of the trained models
        # (useful when X may have been used to train some of the
        # models)
        if models is None:
            models = self.models
        else:
            models = [self.models[i] for i in models]

        priors = np.zeros((X.shape[0], 2))
        for model in models:
            model.eval()
            priors += model(autograd.Variable(torch.FloatTensor((X - self.X_means) / self.X_std), requires_grad=False)).data.numpy()
        priors /= float(len(models))
        if y is None:
            return priors

        # Get the posterior estimates
        mu0, sigma0 = self.null_dist
        P0 = norm.pdf(y, mu0, sigma0)[:,np.newaxis]
        P1 = self.alt_dist.pdf(y)[:,np.newaxis]
        prior_grid = beta.pdf(self.beta_grid, priors[:,0:1], priors[:,1:2])
        prior_grid /= prior_grid.sum(axis=1, keepdims=True)
        post0 = P0 * (1-self.beta_grid)
        post1 = P1 * self.beta_grid
        posteriors = ((post1 / (post0 + post1)) * prior_grid).sum(axis=1)
        posteriors = posteriors.clip(1e-8, 1-1e-8)

        return priors, posteriors












