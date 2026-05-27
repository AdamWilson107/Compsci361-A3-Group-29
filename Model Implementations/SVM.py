
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "train.csv"
TEST_PATH = PROJECT_ROOT / "test.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "svm"
RANDOM_STATE = 29

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / "matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / "cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def load_data():
    """Load train/test CSV files."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    x_train = train_df["Article"].fillna("")
    y_train = train_df["Category"]
    x_test = test_df["Article"].fillna("")
    y_test = test_df["Category"]

    return x_train, y_train, x_test, y_test


def run_grid_search(x_train, y_train):
    """Run 5-fold CV over SVM kernels and key hyperparameters."""
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("svm", SVC()),
        ]
    )

    param_grid = [
        {
            "svm__kernel": ["linear"],
            "svm__C": [0.01, 0.1, 1, 10, 100],
        },
        {
            "svm__kernel": ["rbf"],
            "svm__C": [0.1, 1, 10, 100],
            "svm__gamma": [0.001, 0.01, 0.1, 1],
        },
    ]

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=1,
        return_train_score=True,
    )
    grid.fit(x_train, y_train)
    return grid


def save_cv_results(grid):
    """Save sorted cross-validation results for the report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cv_results = pd.DataFrame(grid.cv_results_)
    columns = [
        "param_svm__kernel",
        "param_svm__C",
        "param_svm__gamma",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
    ]
    existing_columns = [col for col in columns if col in cv_results.columns]
    cv_summary = cv_results[existing_columns].sort_values(
        "mean_test_score", ascending=False
    )
    cv_summary.to_csv(OUTPUT_DIR / "svm_cv_results.csv", index=False)
    return cv_summary


def evaluate_best_model(grid, x_test, y_test):
    """Evaluate the best CV model on the held-out test set."""
    best_model = grid.best_estimator_
    y_pred = best_model.predict(x_test)
    test_f1 = f1_score(y_test, y_pred, average="macro")

    report = classification_report(y_test, y_pred)
    with open(OUTPUT_DIR / "svm_test_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Best parameters: {grid.best_params_}\n")
        f.write(f"Best CV macro F1: {grid.best_score_:.4f}\n")
        f.write(f"Test macro F1: {test_f1:.4f}\n\n")
        f.write(report)

    return best_model, test_f1, report


def gamma_from_sigma(sigma):
    """Convert RBF kernel width sigma to sklearn gamma."""
    return 1 / (2 * sigma**2)


def plot_decision_boundary(ax, model, x_2d, y, title):
    """Plot a binary SVM decision boundary in an already reduced 2-D space."""
    x_min, x_max = x_2d[:, 0].min() - 0.15, x_2d[:, 0].max() + 0.15
    y_min, y_max = x_2d[:, 1].min() - 0.15, x_2d[:, 1].max() + 0.15
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 180),
        np.linspace(y_min, y_max, 180),
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid_points)
    label_to_int = {label: index for index, label in enumerate(model.classes_)}
    zz_int = np.array([label_to_int[label] for label in zz]).reshape(xx.shape)
    y_int = np.array([label_to_int[label] for label in y])

    ax.contourf(xx, yy, zz_int, alpha=0.25, cmap="coolwarm")
    scatter = ax.scatter(
        x_2d[:, 0],
        x_2d[:, 1],
        c=y_int,
        cmap="coolwarm",
        edgecolor="black",
        linewidth=0.35,
        s=32,
    )
    ax.set_title(title)
    ax.set_xlabel("SVD component 1")
    ax.set_ylabel("SVD component 2")
    ax.legend(
        handles=scatter.legend_elements()[0],
        labels=list(model.classes_),
        title="Category",
        loc="best",
    )


def create_decision_boundary_plots(x_train, y_train):
    """
    TF-IDF features are high-dimensional, so TruncatedSVD projects them into a
    2-D plane for visualization. The final evaluated model still uses the full
    TF-IDF feature space.
    """
    vectorizer = TfidfVectorizer()
    x_tfidf = vectorizer.fit_transform(x_train)
    svd = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
    x_2d = svd.fit_transform(x_tfidf)

    linear_c = 1
    rbf_sigma = 0.5
    rbf_gamma = gamma_from_sigma(rbf_sigma)

    linear_svm = SVC(kernel="linear", C=linear_c)
    rbf_hard_margin_c = 10_000
    rbf_svm = SVC(kernel="rbf", C=rbf_hard_margin_c, gamma=rbf_gamma)

    linear_svm.fit(x_2d, y_train)
    rbf_svm.fit(x_2d, y_train)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    plot_decision_boundary(
        axes[0],
        linear_svm,
        x_2d,
        y_train,
        f"Soft-margin linear SVM (C={linear_c})",
    )
    plot_decision_boundary(
        axes[1],
        rbf_svm,
        x_2d,
        y_train,
        f"Hard-margin RBF SVM (C={rbf_hard_margin_c:g}, sigma={rbf_sigma})",
    )

    fig.suptitle("SVM decision boundaries after TF-IDF + TruncatedSVD")
    fig.savefig(OUTPUT_DIR / "svm_decision_boundaries.png", dpi=200)
    plt.close(fig)


def training_size_experiment(best_params, x_train, y_train, x_test, y_test):
    """Measure train/test macro F1 for different training data sizes."""
    portions = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = []
    n_train = len(x_train)

    for portion in portions:
        size = max(1, int(portion * n_train))
        x_subset = x_train.iloc[:size]
        y_subset = y_train.iloc[:size]

        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("svm", SVC()),
            ]
        )
        model.set_params(**best_params)
        model.fit(x_subset, y_subset)

        train_pred = model.predict(x_subset)
        test_pred = model.predict(x_test)
        results.append(
            {
                "portion": portion,
                "train_size": size,
                "train_f1_macro": f1_score(y_subset, train_pred, average="macro"),
                "test_f1_macro": f1_score(y_test, test_pred, average="macro"),
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "svm_training_size_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(
        results_df["portion"],
        results_df["train_f1_macro"],
        marker="o",
        label="Train macro F1",
    )
    ax.plot(
        results_df["portion"],
        results_df["test_f1_macro"],
        marker="o",
        label="Test macro F1",
    )
    ax.set_xlabel("Training data portion")
    ax.set_ylabel("Macro F1")
    ax.set_title("SVM performance by training-set size")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(OUTPUT_DIR / "svm_training_size_f1.png", dpi=200)
    plt.close(fig)

    return results_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = load_data()

    print("Running 5-fold SVM grid search...", flush=True)
    grid = run_grid_search(x_train, y_train)
    cv_summary = save_cv_results(grid)

    print("\nTop 10 SVM CV results:", flush=True)
    print(cv_summary.head(10).to_string(index=False), flush=True)
    print(f"\nBest parameters: {grid.best_params_}", flush=True)
    print(f"Best CV macro F1: {grid.best_score_:.4f}", flush=True)

    _, test_f1, report = evaluate_best_model(grid, x_test, y_test)
    print(f"\nTest macro F1: {test_f1:.4f}", flush=True)
    print("\nClassification report:", flush=True)
    print(report, flush=True)

    print("Creating SVM decision boundary plots...", flush=True)
    create_decision_boundary_plots(x_train, y_train)

    print("Running training-size experiment...", flush=True)
    size_results = training_size_experiment(
        grid.best_params_, x_train, y_train, x_test, y_test
    )
    print("\nTraining-size results:", flush=True)
    print(size_results.to_string(index=False), flush=True)

    print(f"\nSaved SVM outputs to: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
