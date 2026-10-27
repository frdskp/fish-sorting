#!/usr/bin/env python3
"""Compare three classifiers for the fish sorting dataset."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path(__file__).with_name("fish_data.csv")
LABELS = ("Salmon", "Sea Bass")
LABEL_COLORS = ("#f4a261", "#457b9d")


def load_data(path):
    with path.open(newline="", encoding="utf-8") as data_file:
        rows = list(csv.DictReader(data_file))

    required_columns = ("length_cm", "colour_value", "label")
    if not rows or tuple(rows[0]) != required_columns:
        raise ValueError(f"{path} must contain columns {sorted(required_columns)}")

    try:
        features = np.array(
            [
                [float(row["length_cm"]), float(row["colour_value"])]
                for row in rows
            ],
            dtype=float,
        )
    except ValueError as error:
        raise ValueError(f"{path} contains a non-numeric feature value") from error
    labels = np.array([row["label"] for row in rows])
    unknown_labels = set(labels) - set(LABELS)
    if unknown_labels:
        raise ValueError(f"Unknown sorting labels: {sorted(unknown_labels)}")
    if len(rows) < 200:
        raise ValueError(f"{path} has {len(rows)} rows; at least 200 are required")
    if not np.isfinite(features).all():
        raise ValueError(f"{path} contains non-finite feature values")
    if ((features[:, 0] < 5) | (features[:, 0] > 40)).any():
        raise ValueError("length_cm values must be between 5 and 40")
    if ((features[:, 1] < 0) | (features[:, 1] > 100)).any():
        raise ValueError("colour_value values must be between 0 and 100")
    return features, labels


def make_model(classifier):
    return make_pipeline(StandardScaler(), classifier)


def plot_boundaries(models, features, labels, output_path):
    length_grid, colour_grid = np.meshgrid(
        np.linspace(5, 40, 350),
        np.linspace(0, 100, 350),
    )
    mesh_features = np.column_stack((length_grid.ravel(), colour_grid.ravel()))

    figure, axes = plt.subplots(1, len(models), figsize=(18, 6), sharex=True, sharey=True)
    for axis, (name, model, accuracy, balanced_accuracy) in zip(axes, models):
        predictions = model.predict(mesh_features).reshape(length_grid.shape)
        regions = np.where(predictions == LABELS[0], 0, 1)
        axis.contourf(
            length_grid,
            colour_grid,
            regions,
            levels=np.arange(len(LABELS) + 1) - 0.5,
            cmap=ListedColormap(LABEL_COLORS),
            alpha=0.45,
        )

        for label, colour in zip(LABELS, LABEL_COLORS):
            indices = labels == label
            axis.scatter(
                features[indices, 0],
                features[indices, 1],
                marker="o",
                c=colour,
                edgecolors="black",
                linewidths=0.35,
                s=24,
                alpha=0.85,
                label=label,
            )

        axis.set_title(
            f"{name}\nAccuracy: {accuracy:.1%} | Balanced: {balanced_accuracy:.1%}"
        )
        axis.set_xlabel("Length (cm)")
        axis.set_xlim(5, 40)
        axis.set_ylim(0, 100)
        axis.grid(alpha=0.2)

    axes[0].set_ylabel("Colour value (0 = darkest, 100 = brightest)")
    axes[-1].legend(loc="upper right", fontsize=8)
    figure.suptitle(
        "Fish sorting: Salmon vs Sea Bass decision boundaries",
        fontsize=15,
    )
    figure.text(
        0.5,
        0.01,
        "Background = predicted class; points = synthetic fish observations.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.93))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--knn-k",
        type=int,
        default=100,
        help="number of neighbours for KNN (default: 10)",
    )
    parser.add_argument(
        "--tree-depth",
        type=int,
        default=2,
        help="maximum decision-tree depth (default: 2)",
    )
    parser.add_argument(
        "--linear-c",
        type=float,
        default=0.1,
        help="inverse regularization strength for logistic regression (default: 0.1)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="CSV file with length_cm, colour_value, and label columns",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("fish_decision_boundaries.png"),
        help="path for the output comparison plot",
    )
    args = parser.parse_args()

    if args.knn_k < 1:
        parser.error("--knn-k must be at least 1")
    if args.tree_depth < 1:
        parser.error("--tree-depth must be at least 1")
    if args.linear_c <= 0:
        parser.error("--linear-c must be greater than 0")

    features, labels = load_data(args.data)
    all_indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    if args.knn_k > len(train_indices):
        parser.error(f"--knn-k cannot exceed the training size ({len(train_indices)})")

    classifiers = [
        (
            f"K-nearest neighbours (k={args.knn_k})",
            KNeighborsClassifier(n_neighbors=args.knn_k),
        ),
        (
            f"Linear classifier (logistic, C={args.linear_c:g})",
            LogisticRegression(C=args.linear_c, max_iter=1000),
        ),
        (
            f"Decision tree (max depth={args.tree_depth})",
            DecisionTreeClassifier(max_depth=args.tree_depth, random_state=42),
        ),
    ]
    results = []
    for name, classifier in classifiers:
        model = make_model(classifier)
        model.fit(features[train_indices], labels[train_indices])
        predicted = model.predict(features[test_indices])
        accuracy = accuracy_score(labels[test_indices], predicted)
        balanced_accuracy = balanced_accuracy_score(labels[test_indices], predicted)
        print(
            f"{name}: accuracy={accuracy:.3f}, "
            f"balanced_accuracy={balanced_accuracy:.3f}"
        )
        results.append((name, model, accuracy, balanced_accuracy))

    plot_boundaries(
        results,
        features,
        labels,
        args.output,
    )
    print(f"Decision-boundary comparison saved to {args.output}")


if __name__ == "__main__":
    main()
