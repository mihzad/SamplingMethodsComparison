import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score
from scipy.stats import t

import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
# =========================
# Global Hyperparameters
# =========================

RANDOM_STATE = 42          # Reproducibility
N_REPEATS = 100              # Repeated samplings per sample size
N_POINTS = 40              # Number of evaluated sample sizes
F1_THRESHOLD = 0.90        # Target F1 threshold
CONF_LEVEL = 0.95          # Confidence level
TEST_SIZE = 0.25           # Fixed test set size
USE_PRETRAINED=True              # Whether to train the models or just load results from disk

# =========================
# Data Loading & Preprocessing
# =========================

def load_and_preprocess(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.dropna().reset_index(drop=True)

    if "individual_id" in df.columns:
        df["individual_id"] = df["individual_id"].astype("category").cat.codes

    y = df["species"]
    X = df.drop(columns=["species"])

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

    return X, y, numeric_cols.tolist(), categorical_cols.tolist()


# =========================
# Train / Test Split
# =========================

def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


# =========================
# Models
# =========================

def build_model(model_name, preprocessor):
    match model_name:
        case "Logistic Regression":
            return Pipeline([
                    ("prep", preprocessor),
                    ("clf", LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE
                    ))
                ])
        
        case "KNN":
            return Pipeline([
                    ("prep", preprocessor),
                    ("clf", KNeighborsClassifier(n_neighbors=5))
                ])
        
        case "Random Forest":
            return Pipeline([
                    ("prep", preprocessor),
                    ("clf", RandomForestClassifier(
                        n_estimators=50,
                        class_weight="balanced",
                        random_state=RANDOM_STATE
                    ))
                ])
    


# =========================
# Utilities
# =========================

def generate_sample_sizes(y_train):
    n_classes = y_train.nunique()
    min_n = max(10, n_classes * 3) #minimum adequate sample size:
    # 3 samples per class, but at least 10 samples to start with

    sizes = np.linspace(
        min_n,
        len(y_train),
        N_POINTS,
        dtype=int
    )

    return sizes, min_n


def mean_confidence_interval(values):
    values = np.array(values)
    mean = values.mean()
    std = values.std(ddof=1)

    h = t.ppf(
        (1 + CONF_LEVEL) / 2,
        len(values) - 1
    ) * std / np.sqrt(len(values))

    return mean, mean - h, mean + h


# =========================
# Sampling Strategies
# =========================

def simple_random_sample(X, y, n):
    idx = np.random.choice(len(X), n, replace=False)
    return X.iloc[idx], y.iloc[idx]


def bernoulli_sample(X, y, p):
    mask = np.random.rand(len(X)) < p
    return X[mask], y[mask]


def systematic_sample(X, y, n):
    step = len(X) // n
    start = np.random.randint(0, step)
    idx = np.arange(start, start + step * n, step)
    return X.iloc[idx], y.iloc[idx]


def stratified_sample(X, y, n):

    strata = X["island"].astype(str) + "_" + y.astype(str)

    df_tmp = X.copy()
    df_tmp["y"] = y
    df_tmp["strata"] = strata

    sampled = (
        df_tmp
        .groupby("strata", group_keys=False)
        .apply(
            lambda g: g.sample(
                max(1, int(len(g) / len(df_tmp) * n))  # Proportional allocation with at least 1 sample per stratum
            ),
            include_groups=False   # Fixes DeprecationWarning
        )
    )

    return sampled.drop(columns=["y"]), sampled["y"]


def build_sampler(sampling_name, X_train, y_train):
    match sampling_name:
        case "Simple Random":
            return lambda n: simple_random_sample(X_train, y_train, n)
        
        case "Bernoulli":
            return lambda n: bernoulli_sample(X_train, y_train, n / len(X_train))
        
        case "Systematic":
            return lambda n: systematic_sample(X_train, y_train, n)
        
        case "Stratified":
            return lambda n: stratified_sample(X_train, y_train, n)
        
        case _:
            raise ValueError(f"Unknown sampling method: {sampling_name}")

# =========================
# Core Experiment
# =========================

def run_single_experiment(
    sampling_name, model_name, X_train, y_train, X_test, y_test,
    dataset_info_numeric_cols, dataset_info_categorical_cols,
    sample_sizes, min_n, n_repeats_per_sample
    ):

    sampler = build_sampler(sampling_name, X_train, y_train)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), dataset_info_numeric_cols),
            ("cat", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ), dataset_info_categorical_cols),
        ],
        remainder="drop",
    )

    model = build_model(model_name, preprocessor)

    worker_results = {
        "sample_size": [],
        "f1_mean": [],
        "f1_ci_low": [],
        "f1_ci_high": []
    }

    for n in sample_sizes:
        f1_scores = []

        for _ in range(n_repeats_per_sample):
            X_s, y_s = sampler(n)
            if len(X_s) < min_n:
                continue

            model.fit(X_s, y_s)
            preds = model.predict(X_test)

            f1_scores.append(
                f1_score(y_test, preds, average="macro")
            )
        
        if len(f1_scores) < n_repeats_per_sample // 2: # skip if too few valid samples were obtained.
            continue

        f1_mean, f1_low, f1_high = mean_confidence_interval(f1_scores)

        worker_results["sample_size"].append(n)
        worker_results["f1_mean"].append(f1_mean)
        worker_results["f1_ci_low"].append(f1_low)
        worker_results["f1_ci_high"].append(f1_high)

    return (sampling_name, model_name, worker_results)



def run_experiments(X_train, y_train, X_test, y_test,
                     dataset_info_numeric_cols, dataset_info_categorical_cols, sample_sizes, min_n):
    
    sampling_names = ["Simple Random", "Bernoulli", "Systematic", "Stratified"]
    model_names = ["Logistic Regression", "KNN", "Random Forest"]

    results = {}

    tasks = [
        (s_name, m_name) 
        for s_name in sampling_names 
        for m_name in model_names
    ]

    raw_results = Parallel(n_jobs=-1)(
        delayed(run_single_experiment)(
            s_name, m_name, 
            X_train, y_train, X_test, y_test,
            dataset_info_numeric_cols, dataset_info_categorical_cols,
            sample_sizes, min_n, N_REPEATS
        ) for s_name, m_name in tasks
    )

    results = {}
    for sampling_name, model_name, data in raw_results:
        if sampling_name not in results:
            results[sampling_name] = {}
        results[sampling_name][model_name] = data


    CI_table_dict = {}
    for  model_name in model_names:

        if model_name not in CI_table_dict:
                CI_table_dict[model_name] = {}

        for sampling_name in sampling_names:
            combination_data = results[sampling_name][model_name]

            samples_generated = combination_data["sample_size"]
            f1_means = combination_data["f1_mean"]
            f1_lows = combination_data["f1_ci_low"]
            f1_highs = combination_data["f1_ci_high"]

            #populate CI table for further visualizations
            
            for n, f1_mean, f1_low, f1_high in zip(samples_generated, f1_means, f1_lows, f1_highs):
                status = "REACHED" if f1_mean >= F1_THRESHOLD else "NOT REACHED"
                value = f"{status}. F1 mean: {f1_mean*100:.1f}. F1 CI [{f1_low*100:.1f}, {f1_high*100:.1f}]"

                CI_sampling_table_row_key = n
                CI_sampling_table_column_key = sampling_name

                if CI_sampling_table_row_key not in CI_table_dict[model_name]:
                    CI_table_dict[model_name][CI_sampling_table_row_key] = {}
                
                CI_table_dict[model_name][CI_sampling_table_row_key][CI_sampling_table_column_key] = value

            
    CI_tables = {}
    for model_name in CI_table_dict.keys():
        CI_tables[model_name] = pd.DataFrame.from_dict(CI_table_dict[model_name], orient='index')

    return results, CI_tables

# =========================
# Visualization
# =========================


def visualize_results(results):

    sampling_names = ["Simple Random", "Bernoulli", "Systematic", "Stratified"]
    model_names = ["Logistic Regression", "KNN", "Random Forest"]

    for sampling_name in sampling_names:
        
        plt.figure(figsize=(18, 5))

        for i, model_name in enumerate(model_names, 1):
            combination_data = results[sampling_name][model_name]

            samples_generated = combination_data["sample_size"]
            f1_means = combination_data["f1_mean"]
            f1_lows = combination_data["f1_ci_low"]
            f1_highs = combination_data["f1_ci_high"]

            # Custom x=axis ticks to show only a few key sample sizes and avoid clutter
            #=========================
            log_ticks = np.log2(samples_generated)
            # Determine how many labels we actually want to see
            illustrative_interval_size = 6 #first few points to show in detail
            nonillustrative_interval_size = 4 #number of points to show in the rest of the range
            num_points = len(samples_generated)

            # Generate the indices we want to "keep" and others will be the noise=undrawn.
            if num_points <= illustrative_interval_size:
                show_indices = np.arange(num_points) # Show all if small enough
            else:
                show_indices = np.concatenate([
                    np.arange(illustrative_interval_size),
                    np.linspace(start=illustrative_interval_size*1.5,
                                stop=num_points - 1,
                                num=nonillustrative_interval_size,
                                dtype=int)
                ])

            # Create the tick labels with empty strings for the "noise"
            tick_labels = [
                str(int(s)) if i in show_indices else "" 
                for i, s in enumerate(samples_generated)
            ]
            #=========================

            # Plot
            ax = plt.subplot(1, 3, i)
  
            means_pct = np.array(f1_means) * 100
            lows_pct = np.array(f1_lows) * 100
            highs_pct = np.minimum(np.array(f1_highs), 1.0) * 100

            ax.plot(log_ticks, means_pct)
            ax.fill_between(log_ticks, lows_pct, highs_pct, alpha=0.3)

            ax.set_title(model_name)
            ax.set_xlabel("Sample size (log scale)")
            ax.set_ylabel("F1 score (%)")
            
            ax.set_xticks(log_ticks)
            ax.set_xticklabels(tick_labels, rotation=45)    

            ax.set_ylim(50, 108)
            ax.axhline(F1_THRESHOLD * 100, color="red", linestyle="--", label="F1 Threshold: 90%")
            ax.axhline(100, color="green", linestyle="--", label="100%")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        plt.suptitle(f"{sampling_name} sampling", fontsize=16)
        plt.tight_layout()
        plt.show()

# =========================
# Entry Point
# =========================


if __name__ == "__main__":

    if USE_PRETRAINED and os.path.exists('experiment_results.pkl'):
        print("Loading pre-trained results from disk...")
        results = joblib.load('experiment_results.pkl')
        print("metedata:", results.get('metadata', {}))
        visualize_results(results)


    else:
        X, y, dataset_info_numeric_cols, dataset_info_categorical_cols = load_and_preprocess("penguins.csv")
        X_train, X_test, y_train, y_test = split_data(X, y)
        print("len x_train:", len(X_train))
        print("len x_test:", len(X_test))

        sample_sizes, min_n = generate_sample_sizes(y_train)

        results, CI_tables = run_experiments(
            X_train, y_train,
            X_test, y_test,
            dataset_info_numeric_cols, dataset_info_categorical_cols,
            sample_sizes,
            min_n
        )

        results['metadata'] = {
        'threshold': F1_THRESHOLD,
        'timestamp': datetime.datetime.now(),
        'sample_sizes': sample_sizes
        }
        joblib.dump(results, 'experiment_results.pkl')

        visualize_results(results)

        for model_name, t in CI_tables.items():
            filename = f"f1_threshold_confidence_intervals__{model_name}.csv"
            t.to_csv(filename)

            print(f"\n=== {model_name}: Threshold confidence interval table saved to {filename}")