from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from treeple.tree import ObliqueDecisionTreeClassifier
from treeple.ensemble import ObliqueRandomForestClassifier


def process_seed(j):
    print("Processing seed", j)
    in_bags = pd.read_csv(
        f"/home/hao/ydf/ydf_bags_300/seed_{j}.csv",
        header=None,
    ).to_numpy()

    predict_l = []
    for i, bag in enumerate(in_bags):
        model = ObliqueDecisionTreeClassifier(
            random_state=i,
            feature_combinations=1.5,
            max_features=51,
        )
        model.fit(X.iloc[bag], y.iloc[bag])

        prediction = np.full(X.shape[0], np.nan, dtype=np.float64)
        predic = model.predict(X)
        prediction[~X.index.isin(bag)] = predic[~X.index.isin(bag)]
        predict_l.append(prediction)

    average_oob = np.nanmean(predict_l, axis=0)
    average_label = [int(p >= 0.5) for p in average_oob]
    acc = accuracy_score(average_label, y)

    return acc, np.array(predict_l)


# Load processed wise-1 dataset
df = pd.read_csv("processed_wise1_data.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Run in parallel
results = Parallel(n_jobs=-1)(delayed(process_seed)(j) for j in range(21, 321, 1))

# Unpack
obq_ydf_seed_acc_l, obq_ydf_seed_predict_l = zip(*results)

# --- Save results ---

# 1) Save accuracies into a single CSV
pd.DataFrame({"seed": range(21, 321, 1), "accuracy": obq_ydf_seed_acc_l}).to_csv(
    "result/seed_accuracies.csv", index=False
)

# 2) Save predictions: one CSV per seed
for j, predict_l in enumerate(obq_ydf_seed_predict_l, start=1):
    pd.DataFrame(predict_l).to_csv(
        f"result/ydf_bag/seed_{j}_predictions.csv", index=False
    )