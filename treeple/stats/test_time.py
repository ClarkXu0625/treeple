from treeple.datasets import make_trunk_classification
from sklearn.model_selection import train_test_split
from neofit import NeuroExplainableOptimalFIT
from neofit_ydf import NeuroExplainableOptimalFIT_ydf
#from treeple.stats import NeuroExplainableOptimalFIT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
import pdb



def time_test(dim_range,sample_range):
    time_list = []
    dim_list = []
    sample_list = []
    for dim in dim_range:
        for sample in sample_range:
            X, y = make_trunk_classification(n_samples=sample, n_dim=dim, n_informative=min(dim,600), seed=1)
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=2)
            start_time = time.time()

            # neofit testing, CUDA version
            neofit_cuda = NeuroExplainableOptimalFIT(n_estimators=5000,
                                                    n_permutations=100000,
                                                    clf_type="MORF",
                                                    alpha=0.05, 
                                                    verbose=False, 
                                                    device='cuda')
            p_values1, imp_features1, _ = neofit_cuda.get_significant_features(X_train, y_train)
            end_time1 = time.time()

            # Neofit cpu version
            neofit = NeuroExplainableOptimalFIT(n_estimators=5000,
                                                n_permutations=100000,
                                                clf_type="MORF",
                                                alpha=0.05, 
                                                verbose=False, 
                                                device='cpu')
            p_values2, imp_features2, _ = neofit.get_significant_features(X_train, y_train)
            end_time2 = time.time()

            print(sum(imp_features2.astype(int)-imp_features1.astype(int)))
            # save results of each run
            #os.makedirs("./sex_classification/results/", exist_ok=True)
            #np.save(f"./sex_classification/results/p_values_{sample}_{dim}.npy", p_values)
            #pdb.set_trace()
            time_list.append(end_time1 - start_time)
            #with open("time_log.txt", "a") as log_file:
            #    log_file.write(f"Time taken for sample {sample} and dim {dim} is: {end_time - start_time:.2f} seconds\n")
            dim_list.append(dim)
            sample_list.append(sample)

    # save time_list, dim_list and sample_list
    time_list = np.array(time_list)
    dim_list = np.array(dim_list)
    sample_list = np.array(sample_list)
    # save as a pandas DataFrame
    results_df = pd.DataFrame({
        'time': time_list,
        'dim': dim_list,
        'sample': sample_list
    })
    return results_df


#dim = range(1000,6000,1000)
#dim = np.array([128,256,512,1024,2048])
dim = np.array([1024])
#dim = np.array([124])
sample = dim
#sample = range(1000,6000,1000)
results_time = time_test(dim, sample)
print(results_time)

# save results to a csv file
results_time.to_csv('time_results_fast.csv', index=False)
  