from neofit_test_utils import time_test, plot_runtime_heatmaps, compare_original, time_neofit
import numpy as np


dim_range=np.array([128])
sample_range=dim_range


time_neofit(dim_range, 
        sample_range, 
        clf_type="SPORF", 
        n_permutations=10000,
        n_estimators=5000,
        devices=['cpu', 'cuda'],
        #devices=['cpu_numba','cpu_numba_paired','cpu','cuda','cuda_exact'],
        save_result=False,
        result_path="results/neofit_sporf_all_device_compare.csv")

# compare_original(dim_range, 
#         sample_range, 
#         clf_type="SPORF", 
#         n_permutations=10000,
#         n_estimators=5000,
#         device1='cpu_numba',
#         device2='cpu',
#         save_result=False,
#         result_path="results/neofit_device_compare2.csv")