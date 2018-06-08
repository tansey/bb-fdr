'''Case study of applying BB-FDR to a cancer drug screening.

The data are taken from the Genomics of Drug Sensitivity in Cancer (GDSC). We
preprocessed the max dosage readings and converted them into into z-scores.

BB-FDR performs two stages of selection.

Stage 1: Fit a (black box) neural network model using the mutation data for each
cell line. The NN is then used to select the significant outcomes at a given
false discovery rate (FDR). We use the default 10% FDR.

Stage 2: For each mutation, fit a (black box) gradient boosting classifier to
predict the probability of a mutation given other mutations. The classifier is
then used to perform a conditional randomization test (CRT) to determine
whether that mutation is significantly associated with differential response.

Note that the code is setup to be run easily on a cluster, in case one has
hundreds of such studies to analyze. The model is checkpointed frequently
in order to allow for preemption without losing the progress. The whole process
should finish in about 5-10 minutes on a modern laptop.
'''
from __future__ import print_function
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import sys
import os
import torch
import torch.nn as nn
from two_groups_beta import BlackBoxTwoGroupsModel
from posterior_crt import PosteriorConditionalRandomizationTester
from utils import p_value_2sided, bh_predictions

if __name__ == '__main__':
    # Specify the name of the drug -- either lapatinib or nutlin
    drug = sys.argv[1]

    # File paths where the pytorch models will be saved
    twogroups_model_fname = 'data/cancer/twogroups_{}.pt'.format(drug)
    crt_model_fname = 'data/cancer/crt_{}.pt'.format(drug)

    print('Loading data for drug={}'.format(drug))
    sys.stdout.flush()
    X = np.load('data/cancer/{}_x.npy'.format(drug))
    z = np.load('data/cancer/{}_z.npy'.format(drug)).clip(-10,10) # Clip at +/- 10 since that's effectively zero prob null
    gene_names = np.load('data/cancer/{}_genes.npy'.format(drug))
    cancer_types = np.load('data/cancer/{}_types.npy'.format(drug))

    print('nsamples: {} nfeatures: {}'.format(X.shape[0], X.shape[1]))
    sys.stdout.flush()

    #### Two-groups empirical bayes model ####
    if os.path.exists(twogroups_model_fname):
        print('Found blackbox two-groups model. Loading...')
        fdr_model = torch.load(twogroups_model_fname)
        h_predictions = np.load('data/cancer/{}_h_predictions.npy'.format(drug))
    else:
        print('Creating blackbox 2-groups model')
        fdr_model = BlackBoxTwoGroupsModel(X, z)

        print('Training')
        sys.stdout.flush()
        results = fdr_model.train(save_dir='data/cancer/{}_twogroups'.format(drug),
                                  verbose=True, batch_size=60 if X.shape[0] > 1000 else 10,
                                  num_folds=10, num_epochs=100)

        print('Saving model results')
        torch.save(fdr_model, twogroups_model_fname)

        # Save the Stage 1 significant experimental outcome results
        h_predictions = results['predictions']
        np.save('data/cancer/{}_h_predictions'.format(drug), h_predictions)
        np.save('data/cancer/{}_posteriors'.format(drug), results['posteriors'])
        np.save('data/cancer/{}_priors'.format(drug), results['priors'])

    #### Posterior CRT model ####
    if os.path.exists(crt_model_fname):
        print('Found model. Loading...')
        crt = torch.load(crt_model_fname)
    else:
        print('Model not found. Creating a new one.')
        crt = PosteriorConditionalRandomizationTester(fdr_model)

    for idx in range(X.shape[1]):
        if not np.isnan(crt.p_values[idx]):
            continue
        
        # Run the CRT for this feature
        crt.run(idx, verbose=True, model='gboost',
                     trials_per_feature=1000,
                     p_cutoff=0.2, entropy_tol=0,
                     feature_name='{} (#{})'.format(gene_names[idx],str(idx+1)))

        # Checkpointing in case we're running on a cluster and need to handle preemption
        torch.save(crt, crt_model_fname)

    print('Getting the predictions')
    results = crt.predictions()

    # Save the Stage 2 variable selection results
    np.save('data/cancer/{}_features'.format(drug), results['discoveries'].astype(bool))

    # Print all the significant results in terms of genes and cancer types
    print('No effect found:')
    [print(t) for t,h in zip(cancer_types, h_predictions) if h == 0]
    print('')
    print('Significant treatment effect found:')
    [print(t) for t,h in zip(cancer_types, h_predictions) if h == 1]
    print('')
    print('BB-FDR discoveries: {}'.format(h_predictions.sum()))
    bh_preds = bh_predictions(p_value_2sided(z), 0.1)
    print('BH discoveries:     {}'.format(bh_preds.sum()))
    print('')
    print('Significant genes:')
    for g,pred in zip(gene_names,results['discoveries'].astype(bool)):
        if pred:
            print(g)

    # Plot the results in comparison to BH
    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=3)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        bins = np.linspace(-10,2,30)
        plt.hist(z[~h_predictions.astype(bool)], color='gray', alpha=0.5, label='Null', bins=bins)
        plt.hist(z[h_predictions.astype(bool)], color='orange', alpha=0.7, label='Discoveries', bins=bins)
        plt.xlabel('z', fontsize=36)
        plt.ylabel('Count', fontsize=36)
        plt.savefig('plots/bbfdr-{}.pdf'.format(drug), bbox_inches='tight')
        plt.close()

        plt.hist(z[~bh_preds.astype(bool)], color='gray', alpha=0.5, label='Null', bins=bins)
        plt.hist(z[bh_preds.astype(bool)], color='orange', alpha=0.7, label='Discoveries', bins=bins)
        if drug == 'lapatinib':
            legend_props = {'weight': 'bold', 'size': 28}
            plt.legend(loc='upper left', prop=legend_props)
        plt.xlabel('z', fontsize=36)
        plt.ylabel('Count', fontsize=36)
        plt.savefig('plots/bh-{}.pdf'.format(drug), bbox_inches='tight')
        plt.close()





