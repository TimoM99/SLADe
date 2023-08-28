# Setting maximum number of threads, feel free to change.
import os
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import csv
import argparse
import math
import numpy as np
import pandas as pd
import multiprocessing

from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from multiprocessing import Process

from models.SLADe import SLADe
from models.GP import GP
from models.SSDO import SSDO
from models.HIF import HIF
from models.PSVM import PSVM

import warnings
warnings.filterwarnings("ignore")

def experiment(dataset, noise_pct, N_RANDOM_STATE, N_SAMPLES, LABEL_BUDGET, SAMPLE_SIZE):

    dataset_name = dataset.split('/')[1].split('_')[0]
    ### Open CSV file
    if not os.path.exists('scores/noise_comparison/{}/'.format(noise_pct)):
        os.makedirs('scores/noise_comparison/{}/'.format(noise_pct))
    file = open('scores/noise_comparison/{}/{}_{}p_{}_{}.csv'.format(
        noise_pct,
        dataset_name,
        noise_pct,
        N_RANDOM_STATE,
        N_SAMPLES), 'w')
    results = csv.writer(file)

    for state in range(N_RANDOM_STATE):
        np.random.seed(state)
        ### Load data
        data = arff.loadarff(dataset)
        df = pd.DataFrame(data[0]).drop('id', axis=1)
        df = df.drop_duplicates()

        ### There are some data sets that need special sampling because of their very low contamination factor,
        ### a higher weight to be sampled is given to the anomalies in the dataset.
        special_datasets = ['KDDCup99']

        if dataset_name == 'KDDCup99':
            df = pd.concat([df.loc[df['outlier'] == b'yes'], 
                df.loc[df['outlier'] == b'no'].apply(lambda x: x.sample(4754, random_state=state))], axis=0)
            
        ### Sample the other large datasets to 10 000
        max_size = 5000
        if len(df) > max_size and dataset_name not in special_datasets:
            frac = max_size/len(df)
            df = df.groupby('outlier', group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=state))

        X = df.to_numpy()[:, :-1]
        y = np.array([1 if x == b'yes' else -1 for x in df['outlier'].to_numpy()])

        CONT_FACTOR = np.count_nonzero(y == 1)/np.count_nonzero(y)
        results.writerow(['Dataset size', len(X)])
        results.writerow(['#dimensions', len(X[0])])
        results.writerow(['Contamination factor', CONT_FACTOR])
        file.flush()

        ### Build HA
        pca = PCA(random_state=state).fit(X)
        pca_X = pca.transform(X)
        ha = RandomForestClassifier(class_weight='balanced_subsample', max_depth=5, bootstrap=True, max_samples=0.7, 
            random_state=state, max_features=0.5, n_estimators=500).fit(pca_X, y)
        p = ha.predict_proba(pca_X)[:, 1]
        p = (p - p.min())/(p.max() - p.min())

        # What are the thresholded y values?
        y_ha = np.array([1 if pr >= 0.5 else -1 for pr in p])

        # What is the AUC of the HA against the ground truth y values?
        # Should be 1 or very close to 1 since we're predicting p on training data
        results.writerow(['HA AUC: ', roc_auc_score(y, p)])
        file.flush()

        ###  Add noise
        if noise_pct == 0:
            noisy_p = p
        elif noise_pct == 1.0:
            noisy_p = 1 - p
        else:
            # Select stratified percentage of soft labels to make noisy.
            # Stratify using thresholded y_ha so that as many p > 0.5 swap as p < 0.5
            X_1, X_2, y_1, y_2, p_1, p_2 = train_test_split(X, y_ha, p, test_size=noise_pct, stratify=y_ha, random_state=state)
            X = np.concatenate((X_1, X_2))
            y_ha = np.concatenate((y_1, y_2))
            p = np.concatenate((p_1, p_2))
            noisy_p = np.concatenate((p_1, 1 - p_2))

        results.writerow(['Effective anomaly noise rate HA', np.count_nonzero((p != noisy_p)[y_ha == 1])/np.count_nonzero(y_ha == 1)])
        results.writerow(['Effective normal noise rate HA', np.count_nonzero((p != noisy_p)[y_ha == -1])/np.count_nonzero(y_ha == -1)])
        file.flush()
        ### Start experiment
        # Stratified split over y_ha (p > 0.5 vs p < 0.5)
        X_train, X_test, _, p_test, p_train, _ = train_test_split(X, p, noisy_p, stratify=y_ha, random_state=state, test_size=0.2)

        for sample in range(N_SAMPLES):
            # Keep track of where we are in the loop
            print("Noise pct = {} - Random state = {} - Sample number = {}".format(noise_pct, state, sample))
            print("Label percentage = 0%")
            # Sample hard labels
            y_train = np.array([np.random.binomial(1, pr) for pr in p_train])
            y_train[y_train == 0] = -1
            # We need at least 32 normals in the training data, which is very likely.
            # However when our sampling is really unlucky, we need to make sure that we
            # have more normal hard labels than trees in HIF (32). 
            while np.count_nonzero(y_train == -1) < 32:
                y_train = np.array([np.random.binomial(1, pr) for pr in p_train])
                y_train[y_train == 0] = -1
            
            y_test = np.zeros(len(X_test))
            # We need at least 1 anomaly in the test set, otherwise AUROC does not exist.
            # Since every method is evaluated on the same test set, no method has an advantage.
            while np.count_nonzero(y_test == 1) == 0:
                y_test = np.array([np.random.binomial(1, pr) for pr in p_test])
                y_test[y_test == 0] = -1

            # Keep track of results, to be saved as one row in CSV.
            r = []

            # If we want to compare relative AUROC scores, we keep track of the 'best possible' 
            # AUROC score. We define it as the p generated for the test set against the labels 
            # sampled from it. For thresholded labels, this would be 1, but for sampled labels
            # that is not the case
            r.append(roc_auc_score(y_test, p_test))

            ### Setup unsupervised versions of models.
            models = {
                # SLADe used its default hyperparameters here as defined in the paper.
                'SLADe': SLADe(CONT_FACTOR, state),
                'SSDO': SSDO(CONT_FACTOR, state),
                'P-SVM': PSVM(state),
                'GP': GP(state),
                'HIF': HIF(CONT_FACTOR, state)
                }
            
            if sample == 0 and state == 0:
                # For plotting the results, we keep track of the model order and save it in the CSV.
                results.writerow(models.keys())

            # Fit the models on empty dataset and save scores
            for key in models:
                models[key].fit(X_train, np.zeros(len(y_train)), p_train)
                r.append(roc_auc_score(y_test, models[key].predict_proba(X_test)[:, 1]))

            ### Save sampled labels for every method, initialise it randomly -> for P-SVM and GP as they can't make predictions without any labels.
            ### Others are initialised by unsupervised confidence metric ExCeeD (https://github.com/Lorenzo-Perini/Confidence_AD/blob/master/ExCeeD.py)
            ### 1 means labeled as anomaly, -1 as normal and 0 is unlabeled
            n_samples = math.floor(SAMPLE_SIZE * len(X_train))
            labeled_data_init = np.zeros(len(X_train))
            
            labeled_data = dict()
            for k in models:
                if k == 'SLADe':
                    index_to_label = [x[1] for x in list(sorted(zip(models[k].prior_model.predict_confidence(X_train), list(range(len(X_train)))), key=lambda pair:pair[0]))]
                    index_to_label = index_to_label[:n_samples]
                elif k == 'SSDO':
                    index_to_label = [x[1] for x in list(sorted(zip(models[k].ssdo.base_detector.predict_confidence(X_train), list(range(len(X_train)))), key=lambda pair:pair[0]))]
                    index_to_label = index_to_label[:n_samples]
                elif k == 'HIF':
                    index_to_label = [x[1] for x in list(sorted(zip(models[k].predict_confidence(X_train), list(range(len(X_train)))), key=lambda pair:pair[0]))]
                    index_to_label = index_to_label[:n_samples]
                else:
                    np.random.seed(state)
                    index_to_label = np.random.choice(len(X_train), size=n_samples)

                l = labeled_data_init.copy()
                l[index_to_label] = y_train[index_to_label]
                labeled_data[k] = l

            for k in models:
                models[k].fit(X_train, labeled_data[k], p_train)
                r.append(roc_auc_score(y_test, models[k].predict_proba(X_test)[:, 1]))
            
            ### Start active learning loop
            for sample_i in range(1, round(LABEL_BUDGET/SAMPLE_SIZE)):
                print("Label percentage = {}%".format(int(sample_i*SAMPLE_SIZE*100)))
                n_samples = math.floor((sample_i + 1) * SAMPLE_SIZE * len(X_train)) - len(np.where(labeled_data[list(models.keys())[0]] != 0)[0])
                for k in models:
                    models[k].sample(X_train, n_samples, y_train, labeled_data[k])
                    models[k].fit(X_train, labeled_data[k], p_train)
                    r.append(roc_auc_score(y_test, models[k].predict_proba(X_test)[:, 1]))
            results.writerow(r)
            file.flush()
                
    file.close()


if __name__ == "__main__":
    # Parallelized the code, but one can unparallelize it.
    print("Number of CPU scores = {}".format(multiprocessing.cpu_count()))
    ### Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', required=True)
    parser.add_argument('-n_random_states', required=True)
    parser.add_argument('-n_samples', required=True)
    parser.add_argument('-label_budget', default=0.6)
    parser.add_argument('-sample_size', default=0.05)

    args = parser.parse_args()
    dataset = args.dataset
    N_RANDOM_STATE = int(args.n_random_states)
    N_SAMPLES = int(args.n_samples)
    LABEL_BUDGET = float(args.label_budget)
    SAMPLE_SIZE = float(args.sample_size)
                    
    processes = []
    for noise_pct in [0.0, 0.1, 0.2]:
        proc = Process(target=experiment, args=(dataset, noise_pct, N_RANDOM_STATE, N_SAMPLES, LABEL_BUDGET, SAMPLE_SIZE))
        processes.append(proc)

        proc.start()

    for proc in processes:
        proc.join()
                
                


