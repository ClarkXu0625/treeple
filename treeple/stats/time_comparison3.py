# profile_neofit.py
import cProfile
import pstats
import io
from neofit import NeuroExplainableOptimalFIT
from sklearn.datasets import make_classification

def run():
    X, y = make_classification(n_samples=256, n_features=256, n_informative=20, random_state=42)
    model = NeuroExplainableOptimalFIT(
        n_estimators=100,
        n_permutations=100,
        clf_type="SPORF",
        alpha=0.05,
        verbose=1
    )
    model.feat_imp_test(X, y)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    run()
    profiler.disable()

    stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
    stats.print_stats(30)  # Top 30 slowest functions
