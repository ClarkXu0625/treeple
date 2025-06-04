from treeple.datasets import make_trunk_classification
from sklearn.model_selection import train_test_split
#from profit import PermutateRankingOFIT
from treeple.stats import NeuroExplainableOptimalFIT
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import tqdm
import joblib
import shap
import lime

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

def time_test(dim_range,sample_range):
    time_list_profit = []
    time_list_shap = []
    time_list_lime = []
    time_list_permutation = []
    for dim in dim_range:
        for sample in sample_range:
            X, y = make_trunk_classification(n_samples=sample, n_dim=dim, n_informative=min(dim,600), seed=0)
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)
            
            # profit testing
            print(f"profit testing on dim: {dim}, sample: {sample}")
            start_time = time.time()
            profit = NeuroExplainableOptimalFIT(n_estimators=5000,n_permutations=100000,clf_type="SPORF",alpha=0.05, use_oob_impurity=False,verbose=True)
            p_values, imp_features, _ = profit.get_significant_features(X_train, y_train)
            np.save(f"./sex_classification/results/p_values_{sample}_{dim}.npy", p_values)
            end_time = time.time()
            time_list_profit.append(end_time - start_time)

            # save the random forest model
            print(f"train random forest on dim: {dim}, sample: {sample}")
            start_time_rf = time.time()
            rf = RandomForestClassifier(n_estimators=5000, random_state=0)
            rf.fit(X_train, y_train)
            joblib.dump(rf, f"./sex_classification/models/rf_{sample}_{dim}.pkl")
            end_time_rf = time.time()

            # SHAP explain the model
            print(f"SHAP explain the model on dim: {dim}, sample: {sample}")
            start_time_shap = time.time()
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_val)
            np.save(f"./sex_classification/results_SHAP_testing/shap_values_{sample}_{dim}.npy", shap_values)
            end_time_shap = time.time()
            time_list_shap.append(end_time_shap - start_time_shap+end_time_rf - start_time_rf)
            print(f"Time taken for shap: {end_time_shap - start_time_shap+end_time_rf - start_time_rf} seconds")

            # LIME explain the model
            print(f"LIME explain the model on dim: {dim}, sample: {sample}")
            start_time_lime = time.time()
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train,
                mode='classification',
                feature_names=list(range(dim))  # Convert range to list to ensure compatibility
            )
            lime_exp = explainer.explain_instance(
                X_val[0].reshape(1, -1).flatten(),  # Use the specific sample from X_val
                rf.predict_proba,
                num_features=dim
            )
            lime_values = lime_exp.as_list()
            importance_scores = np.array([value for _, value in lime_values])
            np.save(f"./sex_classification/results_LIME_testing/lime_values_{sample}_{dim}.npy", importance_scores)
            end_time_lime = time.time()
            time_list_lime.append(end_time_lime - start_time_lime+end_time_rf - start_time_rf)
            print(f"Time taken for lime: {end_time_lime - start_time_lime+end_time_rf - start_time_rf} seconds")

            # permutation test using sklearn
            print(f"permutation test on dim: {dim}, sample: {sample}")
            start_time_permutation = time.time()
            imp_features = permutation_importance(rf, X_val, y_val, n_repeats=1, scoring="accuracy",n_jobs=-1,random_state=0)
            np.save(f"./sex_classification/results_permutation_testing/imp_features_{sample}_{dim}.npy", imp_features)
            end_time_permutation = time.time()
            time_list_permutation.append(end_time_permutation - start_time_permutation+end_time_rf - start_time_rf)
            print(f"Time taken for permutation: {end_time_permutation - start_time_permutation+end_time_rf - start_time_rf} seconds")


            # time_list.append(end_time - start_time)
            

    return time_list_profit, time_list_shap, time_list_lime, time_list_permutation


dim = np.array([128, 256, 512, 1024, 2048, 4096, 8192])
time_list_profit, time_list_shap, time_list_lime, time_list_permutation = time_test(dim,dim)
np.save("./sex_classification/results_permutation_testing/time_list_profit.npy", time_list_profit)
np.save("./sex_classification/results_permutation_testing/time_list_shap.npy", time_list_shap)
np.save("./sex_classification/results_permutation_testing/time_list_lime.npy", time_list_lime)
np.save("./sex_classification/results_permutation_testing/time_list_permutation.npy", time_list_permutation)
  