
from sklearn.model_selection import train_test_split
import tqdm
import joblib
import shap
import lime
import os
import sys

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from neofit import NeuroExplainableOptimalFIT
from treeple.datasets import make_trunk_classification
#from ..neofit import NeuroExplainableOptimalFIT
#from treeple.stats import NeuroExplainableOptimalFIT
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import pandas as pd

# n_samples = 1000
# n_dim = 784
# X, y = make_trunk_classification(n_samples=n_samples, n_dim=n_dim, n_informative=600, seed=0)
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)



# print(f"imp_features: {imp_features}")
# np.save("./sex_classification/results/imp_features.npy", imp_features)
# np.save("./sexclassification/results/p_values.npy", p_values)

# plot the p-values
# plt.figure(figsize=(10, 6))
# plt.plot(p_values)  
# plt.xlabel("Feature Index")
# plt.ylabel("P-value")
# plt.title("P-values for Feature Importance")
# plt.show()
# plt.savefig("./sex_classification/results/p_values.png")

# plot the importance features
# plt.figure(figsize=(10, 6))
# plt.bar(imp_features, p_values)
# plt.xlabel("Feature Index")
# plt.ylabel("Importance")
# plt.title("Importance Features")
# plt.show()
def time_test(dim_range, sample_range):
    time_list_profit = []
    time_list_shap = []
    time_list_lime = []
    #time_list_permutation = []

    for dim in dim_range:
        for sample in sample_range:
            X, y = make_trunk_classification(n_samples=sample, n_dim=dim, n_informative=min(dim, 600), seed=0)
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

            # PROFIT
            print(f"profit testing on dim: {dim}, sample: {sample}")
            start_time = time.time()
            profit = NeuroExplainableOptimalFIT(n_estimators=5000, n_permutations=100000, clf_type="SPORF", alpha=0.05, verbose=False)
            p_values, imp_features, _ = profit.get_significant_features(X_train, y_train)
            end_time = time.time()
            os.makedirs("./sex_classification/results", exist_ok=True)
            np.save(f"./sex_classification/results/p_values_{sample}_{dim}.npy", p_values)            
            time_list_profit.append(end_time - start_time)
            print(f"Time taken for profit: {end_time - start_time} seconds")

            # Random Forest
            print(f"train random forest on dim: {dim}, sample: {sample}")
            start_time_rf = time.time()
            rf = RandomForestClassifier(n_estimators=5000, random_state=0)
            rf.fit(X_train, y_train)
            #os.makedirs("./sex_classification/models", exist_ok=True)
            #joblib.dump(rf, f"./sex_classification/models/rf_{sample}_{dim}.pkl")
            end_time_rf = time.time()

            # SHAP
            print(f"SHAP explain the model on dim: {dim}, sample: {sample}")
            start_time_shap = time.time()
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_val)
            end_time_shap = time.time()
            os.makedirs("./sex_classification/results_SHAP_testing", exist_ok=True)
            np.save(f"./sex_classification/results_SHAP_testing/shap_values_{sample}_{dim}.npy", shap_values)
            time_list_shap.append(end_time_shap - start_time_shap + end_time_rf - start_time_rf)
            print(f"Time taken for shap: {end_time_shap - start_time_shap + end_time_rf - start_time_rf} seconds")

            # LIME
            print(f"LIME explain the model on dim: {dim}, sample: {sample}")
            start_time_lime = time.time()
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train,
                mode='classification',
                feature_names=list(range(dim))
            )
            lime_exp = explainer.explain_instance(
                X_val[0].reshape(1, -1).flatten(),
                rf.predict_proba,
                num_features=dim
            )
            lime_values = lime_exp.as_list()
            importance_scores = np.array([value for _, value in lime_values])
            end_time_lime = time.time()
            os.makedirs("./sex_classification/results_LIME_testing", exist_ok=True)
            np.save(f"./sex_classification/results_LIME_testing/lime_values_{sample}_{dim}.npy", importance_scores)  
            time_list_lime.append(end_time_lime - start_time_lime + end_time_rf - start_time_rf)
            print(f"Time taken for lime: {end_time_lime - start_time_lime + end_time_rf - start_time_rf} seconds")

            with open("time_log.txt", "a") as log_file:
                log_file.write(f"Time taken for sample {sample} and dim {dim} \n")
                log_file.write(f"neofit: {end_time - start_time:.2f} seconds\n")
                log_file.write(f"shap: {end_time_shap - start_time_shap + end_time_rf - start_time_rf:.2f} seconds\n")
                log_file.write(f"lime: {end_time_lime - start_time_lime + end_time_rf - start_time_rf:.2f} seconds\n")
            # sklearn permutation
            # print(f"permutation test on dim: {dim}, sample: {sample}")
            # start_time_permutation = time.time()
            # imp_features = permutation_importance(rf, X_val, y_val, n_repeats=1, scoring="accuracy", n_jobs=-1, random_state=0)
            # end_time_permutation = time.time()
            # os.makedirs("./sex_classification/results_permutation_testing", exist_ok=True)
            # np.save(f"./sex_classification/results_permutation_testing/imp_features_{sample}_{dim}.npy", imp_features)
            
            # time_list_permutation.append(end_time_permutation - start_time_permutation + end_time_rf - start_time_rf)
            # print(f"Time taken for permutation: {end_time_permutation - start_time_permutation+end_time_rf - start_time_rf} seconds")


    return time_list_profit, time_list_shap, time_list_lime     #, time_list_permutation


dim1 = np.array([256, 512, 1024, 2048, 4096, 8192])
dim2 = dim1
# dim1 = [100]
# dim2 = [100]
dim_vals = dim1
sample_vals = dim2
index = pd.MultiIndex.from_product([dim_vals, sample_vals], names=["dim", "sample"])


time_list_profit, time_list_shap, time_list_lime = time_test(dim1,dim2)

df_profit = pd.DataFrame({"time": time_list_profit}, index=index).unstack(level=-1)
df_shap = pd.DataFrame({"time": time_list_shap}, index=index).unstack(level=-1)
df_lime = pd.DataFrame({"time": time_list_lime}, index=index).unstack(level=-1)
#df_permutation = pd.DataFrame({"time": time_list_permutation}, index=index).unstack(level=-1)

os.makedirs("./sex_classification/results_permutation_testing", exist_ok=True)

# Save to CSV
df_profit.to_csv("./sex_classification/results_permutation_testing/time_list_profit.csv")
df_shap.to_csv("./sex_classification/results_permutation_testing/time_list_shap.csv")
df_lime.to_csv("./sex_classification/results_permutation_testing/time_list_lime.csv")
#df_permutation.to_csv("./sex_classification/results_permutation_testing/time_list_permutation.csv")
