import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from treeple.datasets import make_trunk_classification
import ydf
import matplotlib.pyplot as plt
from treeple import ObliqueRandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from treeple._lib.sklearn.tree._criterion import Gini
from treeple.tree._oblique_splitter import BestObliqueSplitterTester
from treeple.datasets import make_trunk_classification
import pandas as pd
import math
from matplotlib.colors import LogNorm
import cProfile
import pstats
import io
import re
import numpy as np
from scipy.sparse import lil_matrix


def profiling_fit(n_estimators, n_dim, n_samples, max_features, feature_combinations, max_depth, n_jobs, max_leaf_nodes, treeple_params=None):
    '''
    Profile treeple fit function, return profile results
    '''
    
    X, y = make_trunk_classification(n_samples=n_samples, n_dim=n_dim)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    if treeple_params is None:
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_features": max_features,
            "feature_combinations": feature_combinations,
            "n_jobs": n_jobs,
            "max_leaf_nodes": max_leaf_nodes
        }
    else:
        params = treeple_params.copy()
        params.update({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_features": max_features,
            "feature_combinations": feature_combinations,
            "n_jobs": n_jobs,
            "max_leaf_nodes": max_leaf_nodes,
        })

    model = ObliqueRandomForestClassifier(**params)

    
    profiler = cProfile.Profile()
    profiler.enable()

    model.fit(X_train, y_train)

    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
    ps.print_stats()

    return s.getvalue(), 



def constant_nNonzeros_simulation_treeple(n_tree,                   
                                params_treeple,
                                target_non_zeros_per_row,
                                n_rows,
                                n_columns,
                                n_samples=2000, 
                                n_rep=2,
                                plot=False):

    '''
    Trunk simulation for treeple package that keeps constant number of 
    nonzeros in projection matrix rows.
    '''

    accs_treeple = np.zeros((len(n_columns), len(n_rows)))
    times_treeple = np.zeros(accs_treeple.shape)

    params_treeple1 = params_treeple.copy() 
    params_treeple1["n_estimators"] = n_tree
    

    for i, n_column in enumerate(n_columns):
        # n_column matches number of features
        n_dim= n_column


        X, y = make_trunk_classification(n_samples=n_samples, n_dim=n_dim, n_informative=600, seed=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        for j, n_row in enumerate(n_rows):
            # n_rows matches number of projections            
            feature_combination = target_non_zeros_per_row
            params_treeple1["max_features"] = n_row
            params_treeple1["feature_combinations"] = feature_combination
  
            # see if matches the target non-zeros
            _, treeple_n_nonzeros = plot_proj_existing_data(X, y,
                max_features=n_row, 
                feature_combinations=feature_combination,
                plot_fig=False,
                random_state=1)

            print("-----------------------------------------")        
            print(f"Constants: n_dim: {n_dim} | n_tree: {n_tree} | n_samples: {n_samples} | target_non_zeros per row: {target_non_zeros_per_row} | n_rep: {n_rep}")
            print(f"Projection matrix: n_row: {n_row} | n_column: {n_column} | treeple_non_zeros: {treeple_n_nonzeros}")
            print(f"Feature combinations: {feature_combination} | max_features: {n_row}")



            acc_temp_treeple=0
            time_temp_treeple=0

            #f1_temp=0

            for _ in range(n_rep):
            
                # --- Train Treeple ---
                treeple_model = ObliqueRandomForestClassifier(**params_treeple1)
                start_time = time.time()
                treeple_model.fit(X_train, y_train)
                time_treeple = time.time() - start_time

                pred_treeple = treeple_model.predict(X_test)
                if isinstance(pred_treeple[0], np.ndarray):  # Some models return probabilities
                    pred_treeple = np.argmax(pred_treeple, axis=1)

                acc_treeple = accuracy_score(y_test, pred_treeple)
                acc_temp_treeple+=acc_treeple
                time_temp_treeple+=time_treeple           

            accs_treeple[i,j] = acc_temp_treeple/n_rep
            times_treeple[i,j] = time_temp_treeple/n_rep

    return accs_treeple, times_treeple


def constant_nNonzeros_simulation(n_tree,                   
                                params_treeple, 
                                params_ydf,
                                target_non_zeros,
                                n_rows,
                                n_columns,
                                n_samples=2000, 
                                n_rep=2,
                                plot=False):
    '''
    Trunk simulation for treeple and ydf that keeps constant number of 
    nonzeros in projection matrix rows.
    '''


    accs_ydf = np.zeros((len(n_columns), len(n_rows)))
    times_ydf = np.zeros(accs_ydf.shape)
    accs_treeple = np.zeros(accs_ydf.shape)
    times_treeple = np.zeros(accs_ydf.shape)
    f1_scores = np.zeros(accs_ydf.shape)
    ydf_estimator_info = np.empty((len(n_columns), len(n_rows)), dtype=object)
    treeple_estimator_info = np.empty((len(n_columns), len(n_rows)), dtype=object)
    

    # copy the params to avoid overwriting
    params_treeple1 = params_treeple.copy() 
    params_ydf1 = params_ydf.copy()
    params_ydf1["num_trees"] = n_tree
    params_treeple1["n_estimators"] = n_tree
    

    for i, n_column in enumerate(n_columns):
        # n_column matches number of features
        n_dim= n_column


        X, y = make_trunk_classification(n_samples=n_samples, n_dim=n_dim, n_informative=600, seed=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        for j, n_row in enumerate(n_rows):
            # n_rows matches number of projections            
            feature_combination = target_non_zeros / n_row
            params_treeple1["max_features"] = n_row
            params_treeple1["feature_combinations"] = feature_combination
            
            params_ydf1["sparse_oblique_max_num_projections"] = int(n_row)
            params_ydf1["sparse_oblique_projection_density_factor"] = feature_combination
            #params_ydf1["sparse_oblique_num_projections_exponent"] = math.log(n_column, n_row) + 1.0
            params_ydf1["sparse_oblique_num_projections_exponent"] = math.log(n_row, n_column)

            # see if matches the target non-zeros
            _, treeple_n_nonzeros = plot_proj_existing_data(X, y,
                max_features=n_row, 
                feature_combinations=feature_combination,
                plot_fig=False,
                random_state=1)

            print("-----------------------------------------")        
            print(f"Constants: n_dim: {n_dim} | n_tree: {n_tree} | n_samples: {n_samples} | target_non_zeros: {target_non_zeros} | n_rep: {n_rep}")
            print(f"Projection matrix: n_row: {n_row} | n_column: {n_column} | treeple_non_zeros: {treeple_n_nonzeros}")
            print(f"Feature combinations: {feature_combination} | max_features: {n_row}")


            print("------------------------------------------")
            print("YDF parameters:")
            exponent = params_ydf1["sparse_oblique_num_projections_exponent"]
            density_factor = params_ydf1["sparse_oblique_projection_density_factor"]

            num_projection = min(n_row, int(math.ceil(n_column**exponent)+0.5))
            projection_density = density_factor/ n_column
            ydf_nonzeros = num_projection * projection_density * n_column
            print(f"exponent: {exponent} | density_factor: {density_factor} | max_num_projection: {num_projection} | projection_density: {projection_density:.2f}")
            print(f"num of projections: {num_projection} | expected non-zeros: {ydf_nonzeros:.2f}")
            
            acc_temp_ydf=0
            time_temp_ydf=0

            acc_temp_treeple=0
            time_temp_treeple=0

            f1_temp=0

            for _ in range(n_rep):
                # --- Train YDF ---
                acc_ydf, time_ydf, pred_ydf, ydf_model = train_ydf(X_train, y_train, X_test, y_test, params_ydf1)
                acc_temp_ydf+=acc_ydf
                time_temp_ydf+=time_ydf
            
                # --- Train Treeple ---
                treeple_model = ObliqueRandomForestClassifier(**params_treeple1)
                start_time = time.time()
                treeple_model.fit(X_train, y_train)
                time_treeple = time.time() - start_time

                pred_treeple = treeple_model.predict(X_test)
                if isinstance(pred_treeple[0], np.ndarray):  # Some models return probabilities
                    pred_treeple = np.argmax(pred_treeple, axis=1)

                acc_treeple = accuracy_score(y_test, pred_treeple)
                acc_temp_treeple+=acc_treeple
                time_temp_treeple+=time_treeple

                # Calculate F1 score for two predictions
                f1_compare = f1_score(pred_ydf, pred_treeple)
                f1_temp+=f1_compare
            
            ##### store estimator results #####
            # treeple_estimator_info[i,j] = get_treeple_tree_info(treeple_model.estimators_[0])
            # max_depth, total_nodes, oblique_nodes = get_ydf_tree_info(ydf_model.get_tree(0).root)
            # ydf_estimator_info[i,j] = (max_depth, total_nodes, oblique_nodes, total_nodes-oblique_nodes)
            treeple_avg, ydf_avg = check_average_info(treeple_model, ydf_model, n_tree)
            treeple_estimator_info[i,j] = treeple_avg
            ydf_estimator_info[i,j] = ydf_avg

            # check average number of non-zeros per vector
            print("-----------------------------------------")
            _,_,_,avg_non_zeros = extract_avg_projection(ydf_model, n_tree, n_dim)
            print(f"Average non-zeros per projection matrix: {avg_non_zeros*n_row:.2f}")


            # Store results
            accs_ydf[i,j] = acc_temp_ydf/n_rep
            times_ydf[i,j] = time_temp_ydf/n_rep

            accs_treeple[i,j] = acc_temp_treeple/n_rep
            times_treeple[i,j] = time_temp_treeple/n_rep

            f1_scores[i,j] = f1_temp/n_rep

    return accs_ydf, times_ydf, accs_treeple, times_treeple, f1_scores, treeple_estimator_info, ydf_estimator_info



def check_average_info(treeple_model, ydf_model, n_estimator):
    '''
    Check the average number of oblique nodes and leaf nodes in the treeple and ydf models.
    '''
    treeple_estimator_info = np.zeros((n_estimator,4))
    ydf_estimator_info = np.zeros((n_estimator,4))
    
    for i in range(n_estimator):
        treeple_info = get_treeple_tree_info(treeple_model.estimators_[i])
        max_depth, total_nodes, oblique_nodes = get_ydf_tree_info(ydf_model.get_tree(0).root)
        ydf_info = (max_depth, total_nodes, oblique_nodes, total_nodes-oblique_nodes)
        for j in range(4):
            treeple_estimator_info[i,j] = treeple_info[j]
            ydf_estimator_info[i,j] = ydf_info[j]
    treeple_estimator_info = np.mean(treeple_estimator_info, axis=0)
    ydf_estimator_info = np.mean(ydf_estimator_info, axis=0)
    # print("*Treeple estimator info*")
    # print(treeple_estimator_info)

    # print("*YDF estimator info*")
    # print(ydf_estimator_info)
    return treeple_estimator_info, ydf_estimator_info
        
def plot_proj_existing_data(X, y,
                max_features=10, 
                feature_combinations=1.5,
                plot_fig=True,
                random_state=1):
    '''
    Plot the projection matrix from treeple for existing data
    '''

    criterion = Gini(1, np.array((0, 1)))

    min_samples_leaf = 1
    min_weight_leaf = 0.0
    random_state = np.random.RandomState(random_state)
    n_samples= X.shape[0]
    n_features= X.shape[1]

    #feature_combinations = 3.0
    monotonic_cst = None
    missing_value_feature_mask = None

    # X, y = make_trunk_classification(n_samples=n_samples, n_dim=n_features, n_informative=600, seed=0)
    y = y.reshape(-1,1).astype(np.float64)
    X= X.astype(np.float32)

    sample_weight = np.ones(n_samples)

    splitter = BestObliqueSplitterTester(
        criterion,
        max_features,
        min_samples_leaf,
        min_weight_leaf,
        random_state,
        monotonic_cst,
        feature_combinations,
    )
    splitter.init_test(X, y, sample_weight, missing_value_feature_mask)


    projection_matrix = splitter.sample_projection_matrix_py()
    


    if plot_fig:
        # Visualize the projection matrix
        cmap = ListedColormap(["orange", "white", "green"])

        # Create a heatmap to visualize the indices
        fig, ax = plt.subplots(figsize=(6, 6))

        ax.imshow(projection_matrix, cmap=cmap, aspect=n_features / max_features, interpolation="none")

        ax.set(title="Sampled Projection Matrix", xlabel="Feature Index", ylabel="Projection Vector Index")
        ax.set_xticks(np.arange(n_features))
        ax.set_yticks(np.arange(max_features))
        ax.set_yticklabels(np.arange(max_features, dtype=int) + 1)
        ax.set_xticklabels(np.arange(n_features, dtype=int) + 1)

        # Create a mappable object
        sm = ScalarMappable(cmap=cmap)
        sm.set_array([])  # You can set an empty array or values here

        # Create a color bar with labels for each feature set
        colorbar = fig.colorbar(sm, ax=ax, ticks=[0, 0.5, 1], format="%d")
        colorbar.set_label("Projection Weight")
        colorbar.ax.set_yticklabels(["-1", "0", "1"])

        plt.show()
    print("===========================================================================")
    print("projection matrix shape:", projection_matrix.shape)
    print("Created projection matrix with the following parameters:")
    print(f"n_feature = ", n_features, 
          "\nn_samples = ", n_samples, 
          "\nmax_features = ", max_features, 
          "\nfeature_combinations = ", feature_combinations)
    print("max_features * feature_combinations = ", max_features * feature_combinations)
    print("Number of non-zeros: ",len(projection_matrix.nonzero()[0]))
    return projection_matrix, len(projection_matrix.nonzero()[0])

#### Function to get tree object info from estimators ####
def count_oblique_nodes(node):
    if isinstance(node, ydf.tree.Leaf):
        return 0

    count = 0
    if isinstance(node.condition, ydf.tree.NumericalSparseObliqueCondition):
        count += 1

    count += count_oblique_nodes(node.neg_child)
    count += count_oblique_nodes(node.pos_child)
    return count

def get_ydf_tree_info(root_node):
    if isinstance(root_node, ydf.tree.Leaf):
        return 1, 1, 0  # max_depth, num_nodes, num_oblique_nodes

    left_depth, left_nodes, left_oblique = get_ydf_tree_info(root_node.neg_child)
    right_depth, right_nodes, right_oblique = get_ydf_tree_info(root_node.pos_child)

    max_depth = 1 + max(left_depth, right_depth)
    total_nodes = 1 + left_nodes + right_nodes
    oblique_nodes = count_oblique_nodes(root_node) 
    leaf_nodes = total_nodes - oblique_nodes

    return max_depth, total_nodes, oblique_nodes

def get_ydf_model(params_ydf, n_sample=2000, n_estimator=100, n_dim=50, 
                    max_feature=100, feature_combination=3.0, exponent=1.0, X=None, y=None):
    if X is None or y is None:
        X, y = make_trunk_classification(n_samples=n_sample, n_dim=n_dim, n_informative=600, seed=0)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    df_train = prepare_dataframe(X_train, y_train)

    params_ydf1 = params_ydf.copy()
    params_ydf1["num_trees"] = n_estimator
    params_ydf1["sparse_oblique_max_num_projections"] = max_feature
    params_ydf1["sparse_oblique_projection_density_factor"] = feature_combination
    params_ydf1["sparse_oblique_num_projections_exponent"] = exponent


    learner = ydf.RandomForestLearner(**params_ydf1)
    start_time = time.time()
    ydf_model = learner.train(df_train)
    time_taken = time.time() - start_time
    return ydf_model, time_taken

#### treeple ####
def get_treeple_tree_info(treeple_estimator):
    max_depth = treeple_estimator.tree_.max_depth
    total_nodes = treeple_estimator.tree_.node_count
    leaf_nodes = treeple_estimator.tree_.n_leaves
    oblique_nodes = total_nodes - leaf_nodes
    return (max_depth, total_nodes, int(oblique_nodes), int(leaf_nodes))

def get_treeple_model(params_treeple, n_sample=2000, n_estimator=None, n_dim=50, 
                    max_feature=None, feature_combination=None, X=None, y=None):
    if X is None:
        X, y = make_trunk_classification(n_samples=n_sample, n_dim=n_dim, n_informative=600, seed=0)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    params_treeple1 = params_treeple.copy()
    if max_feature is not None:
        params_treeple1["n_estimators"] = n_estimator
    if n_estimator is not None:
        params_treeple1["n_estimators"] = n_estimator
    if feature_combination is not None:
        params_treeple1["feature_combinations"] = feature_combination

    treeple_model = ObliqueRandomForestClassifier(**params_treeple1)
    treeple_model.fit(X_train, y_train)
    treeple_estimators = treeple_model.estimators_
    return treeple_estimators


def prepare_dataframe(X, y):
    '''
    Prepare the dataframe for YDF training.
    '''
    df = pd.DataFrame(X)
    df.columns = [str(i) for i in df.columns]  # Convert column names to strings
    df["target"] = y.astype(int)  # Append target column

    return df


def extract_avg_projection(ydf_model, n_estimator, n_feature):
    '''extract the average projection info from ydf model'''

    non_zeros = np.zeros((n_estimator,))
    proj_dim1 = np.zeros((n_estimator,))
    proj_dim2 = np.zeros((n_estimator,))
    for tree_id in range(n_estimator):
        root_str = str(ydf_model.get_tree(tree_id).root)
        #print("*",root_str)
        projection_matrix, n_nonzeros = selected_weights(root_str, n_feature)

        non_zeros[tree_id] = n_nonzeros
        proj_dim1[tree_id] = projection_matrix.shape[0]
        proj_dim2[tree_id] = projection_matrix.shape[1]

    avg_non_zeros = np.mean(non_zeros)
    avg_proj_dim1 = np.mean(proj_dim1)
    avg_proj_dim2 = np.mean(proj_dim2)
    avg_non_zeros_per_vector = avg_non_zeros / avg_proj_dim1
    #print("Average projection matrix shape:", avg_proj_dim1, avg_proj_dim2)
    print("Average nonzeros number in projection vectors among all nodes:", avg_non_zeros_per_vector)

    return avg_proj_dim1, avg_proj_dim2, avg_non_zeros, avg_non_zeros_per_vector


def selected_weights(tree_str, n_feature):
    """
    Parses a string representation of a YDF decision tree to extract the projection matrix,
    compute its dimension (number of input features), and count non-zero weights.

    Args:
        tree_str (str): Stringified tree structure starting from the root node.

    Returns:
        weight_at_nodes (scipy.sparse.lil_matrix): Sparse matrix of projections.
        proj_dim (int): Dimensionality of each projection (i.e., number of input features).
        num_nonzeros (int): Total number of non-zero weights in the matrix.
    """
    # Find all attribute-weight pairs
    attr_blocks = re.findall(
        r'attributes=\[([0-9,\s]+)\]\s*,\s*weights=\[([0-9eE+.\-,\s]+)\]',
        tree_str
    )
    # print("num of dense vectors: ", len(attr_blocks))

    projection_rows = []
    max_feature_index = -1
    total_nonzeros = 0

    for attr_str, weight_str in attr_blocks:
        attrs = [int(i.strip()) for i in attr_str.split(',') if i.strip()]
        weights = [float(w.strip()) for w in weight_str.split(',') if w.strip()]
        
        if len(attrs) != len(weights):
            raise ValueError(f"Mismatched attributes and weights: {attrs}, {weights}")
        
        total_nonzeros += len(weights)  # Count all weights (assumed non-zero)
        if attrs:
            max_feature_index = max(max_feature_index, max(attrs))
        
        projection_rows.append((attrs, weights))
    #print(projection_rows)

    
    weight_at_nodes = np.zeros((len(attr_blocks), n_feature))

    for i in range(len(projection_rows)):
        attr = projection_rows[i][0]
        #print(attr)
        weight = projection_rows[i][1]
        for j in range(len(weight)):
            weight_at_nodes[i, attr[j]-1] = weight[j]

    return weight_at_nodes, total_nonzeros



def train_ydf(X_train, y_train, X_test, y_test, params_ydf):
    df_train = prepare_dataframe(X_train, y_train)
    df_test = prepare_dataframe(X_test, y_test)

    learner = ydf.RandomForestLearner(**params_ydf)
    start_time = time.time()
    ydf_model = learner.train(df_train)
    time_ydf = time.time() - start_time
    y_pred = ydf_model.predict(df_test)
    y_pred_class = (y_pred >= 0.5).astype(int) 

    acc_ydf = accuracy_score(y_test, y_pred_class)

    # print(f"YDF | n_dim: {n_dim} | n_tree: {params_ydf['num_trees']} | Accuracy: {acc_ydf:.4f} | Train Time: {time_ydf:.4f} sec")
    return acc_ydf, time_ydf, y_pred_class, ydf_model