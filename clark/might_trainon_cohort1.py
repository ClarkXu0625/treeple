from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit
from treeple.ensemble import HonestForestClassifier
from treeple.datasets import (make_trunk_classification,
                              make_trunk_mixture_classification)
from treeple.stats import PermutationHonestForestClassifier, build_oob_forest
from treeple.stats.utils import _mutual_information
from treeple.tree import MultiViewDecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.decomposition import PCA


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import os
from treeple.ensemble import ObliqueRandomForestClassifier
from treeple.tree import  ObliqueDecisionTreeClassifier
from treeple.tree import  PatchObliqueDecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import MDS
# from random import shuffle

import pandas as pd

n_estimators = 100000
MODEL_NAMES = {
    "might": {
        # "n_estimators": n_estimators,
        # "honest_fraction": 0.367,
        # "n_jobs": 40,
        # "bootstrap": True,
        # "stratify": True,
        # "max_samples": 1.6,
        # # "max_features":  'sqrt',
        # # "max_features":  0.3,
        # "honest_prior": "ignore",
        # "honest_method": 'apply',
        # "kernel_method": True,
        # # 'random_state': 80515
    },
        "rf": {
        "n_estimators": int(n_estimators/5),
        "max_features": 'sqrt',
    },
    "knn": {
        # XXX: above, we use sqrt of the total number of samples to allow
        # scaling wrt the number of samples
        # "n_neighbors": 5,
    },
    "svm": {
        "probability": True,
    },
    "lr": {
        "max_iter": 1000,
        "penalty": "l1",
        "solver": "liblinear",
    }
}
might_kwargs = MODEL_NAMES["might"]

filelist = open("/home/ybai31/might/filelist.txt", "r").read().split("\n")[:-1]

# get the sample list
sample_list_file = "/home/ybai31/might/AllSamples.MIGHT.Passed.samples.txt"
sample_list = pd.read_csv(sample_list_file, sep=" ", header=None)
sample_list.columns = ["library", "sample_id", "cohort"]
sample_list.head()
# get the sample_ids where cohort is Cohort1
cohort1 = sample_list[sample_list["cohort"] == "Cohort1"]["sample_id"]
# print(len(cohort1))
cohort2 = sample_list[sample_list["cohort"] == "Cohort2"]["sample_id"]
# print(len(cohort2))
PON = sample_list[sample_list["cohort"] == "PanelOfNormals"]["sample_id"]
# print(cohort1)
sample_list["cohort"].unique()

normal_sample = pd.read_csv('/home/ybai31/might/Normal', sep=" ", header=None) 
normal_sample.columns = ["library", "sample_id"]
print(len(normal_sample))


cancer_sample = pd.read_csv('/home/ybai31/might/Cancer', sep=" ", header=None) 
cancer_sample.columns = ["library", "sample_id"]
cancer_sample_id = cancer_sample ["sample_id"]


def stratified_train_ml(clf,X,y):
    n_samples = X.shape[0]
    cv = StratifiedKFold(n_splits=5, shuffle=True)    
    POS = np.zeros((len(y), 3))
    
    for idx, (train_ix, test_ix) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        
        ### Split Training Set into Fitting Set (40%) and Calibarating Set (40%)
        train_idx = np.arange(
            X_train.shape[0]
        )  # use index array to split, so we can use the same index for the permuted array as well
        fit_idx, cal_idx = train_test_split(
            train_idx, test_size=0.5, random_state=idx, stratify=y_train
        )
        X_fit, X_cal, y_fit, y_cal = (
            X_train[fit_idx],
            X_train[cal_idx],
            y_train[fit_idx],
            y_train[cal_idx],
        )

        
        POS[test_ix, 0] = y_test
        clf.fit(X_fit, y_fit)
        if X_cal.shape[0] <= 1000:
            calibrated_model = CalibratedClassifierCV(
                clf, cv="prefit", method="sigmoid"
            )
        else:
            calibrated_model = CalibratedClassifierCV(
                clf, cv="prefit", method="isotonic"
            )
        calibrated_model.fit(X_cal, y_cal)
        posterior = calibrated_model.predict_proba(X_test)

        POS[test_ix, 1:] = posterior
    return clf,POS



def get_X_y(f, root="/home/hao/comight_real_data/ManuscriptFeatureMatrices/", cohort=cohort2, verbose=False):
    df = pd.read_csv(root + f)
    non_features = ['Run', 'Library', 'Cancer Status', 'Tumor type', 'Stage', 'Library volume (uL)', 'Library Volume', 'UIDs Used', 'Experiment', 'P7', 'P7 Primer', 'MAF']
    sample_ids = df["Sample"]
    sample_ids_type = df['Sample']
    print(sample_ids_type.shape[0])    
    
    for i, sample_id in enumerate(sample_ids):
        if "." in sample_id:
            # print(sample_id.split(".")[1])
            if "Wise" in f or 'ichorCNA' in f:
                sample_ids[i] = sample_id
            else:
                sample_ids[i] = sample_id.split(".")[1]
                
                
    for i, sample_id_type in enumerate(sample_ids_type):
        if "." in sample_id_type:
            # print(sample_id.split(".")[1])
            if "Wise" in f or 'ichorCNA' in f:
                sample_ids_type[i] = sample_id_type
            else:
                sample_ids_type[i] = sample_id_type.split(".")[1]   
                
    target = 'Cancer Status'
    y = df[target]
    # convert the labels to 0 and 1
    y = y.replace("Healthy", 0)
    y = y.replace("Cancer", 1)
    # remove the non-feature columns if they exist
    for col in non_features:
        if col in df.columns:
            df = df.drop(col, axis=1)
    nan_cols = df.isnull().all(axis=0).to_numpy()
    # drop the columns with all nan values
    df = df.loc[:, ~nan_cols]
    # if cohort is not None, filter the samples
    if cohort is not None:
        X = df[(sample_ids.isin(cohort) & sample_ids.isin(sample_ids_type))]
        y = y[(sample_ids.isin(cohort) & sample_ids.isin(sample_ids_type))]
    else:
        X = df
    if "Wise" in f:
        # replace nans with zero
        # print('Wise')
        X = X.fillna(0)
    # impute the nan values with the mean of the column
    X.iloc[:,1:] = X.iloc[:,1:].fillna(X.iloc[:,1:].mean(axis=0))
    # print(X.shape)
    # check if there are nan values
    # nan_rows = X.isnull().any(axis=1)
    nan_cols = X.isnull().all(axis=0)
    # remove the columns with all nan values
    X = X.loc[:, ~nan_cols]
    # print(X.shape)
    if verbose:
        if nan_cols.sum() > 0:
            print(f)
            print(f"nan_cols: {nan_cols.sum()}")
            print(f"X shape: {X.shape}, y shape: {y.shape}")
        else:
            print(f)
            print(f"X shape: {X.shape}, y shape: {y.shape}")
    # X = X.dropna()
    # y = y.drop(nan_rows.index)
    print('num of cancer',np.sum(y))
    return X, y

def run_might(f1,cohort = cohort2,model_name='might',rep = 1):
    # print()
    X_1,y_1 = get_X_y('{}.csv'.format(f1), cohort=cohort, verbose=True)
    X_combine = X_1.iloc[:,1:]
    y = y_1

    X_combine = X_combine.fillna(X_combine.mean(axis=0))
    # k = 0.99
    # pca = PCA(n_components=k)
    # X_combine = pca.fit_transform(X_combine)

    print(f1,X_combine.shape,np.sum(y))
    # scaler = MinMaxScaler()
    # scaler = scaler.fit(X_combine)
    # X_combine = scaler.transform(X_combine)
    if np.sum(y) <= 10:
        print('There is not engough positive y_true')
        return
    
    
    else:
        if model_name == 'might':
            # est = HonestForestClassifier(**might_kwargs)
            # print('MORF')
            est = HonestForestClassifier(
                n_estimators=100000,
                max_samples=1.6,
                max_features = 'sqrt',
                bootstrap=True,
                stratify=True,
                n_jobs=40,
                random_state=515,
                honest_prior="ignore",
                honest_method='apply',
                honest_fraction = 0.367,
                kernel_method = True,
                tree_estimator=ObliqueDecisionTreeClassifier(feature_combinations = 1.5)
                # tree_estimator=PatchObliqueDecisionTreeClassifier(max_patch_dims = np.array([1,3]))
                )
            # perm_est = PermutationHonestForestClassifier(**might_kwargs)
            # covariate_index = np.arange(n_dims_1)
        
        elif model_name == "rf":
            est = RandomForestClassifier( **MODEL_NAMES[model_name],n_jobs = 40)
            # perm_est = RandomForestClassifier( **MODEL_NAMES[model_name],n_jobs = 40)
        elif "knn" in model_name:
            est = KNeighborsClassifier(n_neighbors=int(np.sqrt(X_combine.shape[0]) + 1),)
            # perm_est = KNeighborsClassifier(n_neighbors=int(np.sqrt(X_combine.shape[0]) + 1),)
        elif model_name == "svm":
            est = SVC(**MODEL_NAMES[model_name])
            # perm_est = SVC(**MODEL_NAMES[model_name])
        elif model_name == "lr":
            est = LogisticRegression(**MODEL_NAMES[model_name])
            # perm_est = LogisticRegression(**MODEL_NAMES[model_name])

        if model_name == 'might': 
            est, tree_num = build_oob_forest(
                est,
                np.array(X_combine),
                np.array(y),
                verbose=False,
            )
            
            # perm_est, perm_posterior_arr = build_oob_forest(
            #     perm_est,
            #     np.array(X_combine),
            #     np.array(y),
            #     verbose=False,
            # )
            posterior_arr = tree_num
        else:
            est,posterior_arr = stratified_train_ml(est,np.array(X_combine),np.array(y))
            print(posterior_arr.shape)
        if model_name == 'might':
            # forest_num = np.nanmean(posterior_arr, axis=0)
            # forest_proba_1 = forest_num/ forest_num.sum(axis=1, keepdims=True)
            # print(forest_proba_1.shape)
            # POS = np.nanmean(posterior_arr,axis = 0)
            # POS_perm = np.nanmean(perm_posterior_arr,axis = 0)
            
            forest_num = np.nanmean(tree_num, axis=0)
            forest_proba_kernel = forest_num/ forest_num.sum(axis=1, keepdims=True)
            print(forest_proba_kernel.shape)
            fpr_kernel, tpr_kernel, thresholds_kernel = roc_curve(y, forest_proba_kernel[:,1], pos_label=1,drop_intermediate = False)
            tpr_s_kernel = np.max(tpr_kernel[fpr_kernel<=0.02])
            
            tree_demo = tree_num.sum(axis=-1, keepdims=True)
            tree_proba = tree_num/ tree_demo
            forest_proba = np.nanmean(tree_proba, axis=0)
            fpr, tpr, thresholds = roc_curve(y, forest_proba[:,1], pos_label=1,drop_intermediate = False)
            tpr_s = np.max(tpr[fpr<=0.02])
        else:
            POS = posterior_arr
            # POS_perm = perm_posterior_arr
        
        # fpr, tpr, thresholds = roc_curve(y, POS[:,-1], pos_label=1, drop_intermediate=False,)
        # S98 = np.max(tpr[fpr <= 0.02])
        # tree_depths = [estimator.tree_.max_depth for estimator in est.estimators_]
        # forest_proba = np.nanmean(tree_proba, axis=0)
        
        # fpr, tpr, thresholds = roc_curve(y, forest_proba_1[:,1], pos_label=1,drop_intermediate = False)
        # tpr_s = np.max(tpr[fpr<=0.02])
        print('kernel: ',tpr_s_kernel)
        print('average: ',tpr_s)
        
        tree_depths = [estimator.tree_.max_depth for estimator in est.estimators_]
        print('the tree depths: ', np.mean(tree_depths),np.max(tree_depths),np.min(tree_depths))
        tree_num_nodes = [estimator.tree_.node_count for estimator in est.estimators_]
        print('the # of nodes: ', np.mean(tree_num_nodes),np.max(tree_num_nodes),np.min(tree_num_nodes))
        print(est.feature_importances_)
        scores = est.feature_importances_
        sorted_indices = np.argsort(scores)[::-1]
        # print(sorted_indices[:128])
        # output_fname = ('/home/ybai31/might/might-o-cohort1/'f"pruned_patch1_{model_name}_{f1}_rep{rep}.npz")
        
        # np.savez_compressed(
        #     output_fname,
        #     model_name = model_name,
        #     y=y,
        #     S98_kernel = tpr_s_kernel,
        #     S98_average = tpr_s,
        #     posterior_arr=posterior_arr,
        #     # perm_posterior_arr=perm_posterior_arr,
        #     )
        print(posterior_arr.shape)
        print(model_name,f1,tpr_s_kernel)
        return tpr_s_kernel,tpr_s
# run_might(os.path.splitext(filelist[6])[0],cohort = cohort1,model_name='might',rep = 1) 
Parallel(n_jobs=45)(delayed(run_might)(os.path.splitext(filelist[i])[0],cohort = cohort1,model_name='might',rep = k) 
                    for i in [23]
                    for k in range(1))


