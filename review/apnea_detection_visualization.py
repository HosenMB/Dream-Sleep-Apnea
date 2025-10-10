#!/usr/bin/env python3
"""
Apnea Detection Model Performance Visualization
==============================================

This script creates comprehensive visualizations for the apnea detection model results,
showing performance comparisons across different architectures and ablation studies.

Author: Apnea Detection Analysis Team
Date: 2025-01-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
import ast

warnings.filterwarnings("ignore")

# Set publication-quality style
plt.style.use("default")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "patch.linewidth": 0.5,
    }
)

# Professional color palette
colors = {
    "primary": "#1f77b4",  # Professional blue
    "secondary": "#ff7f0e",  # Professional orange
    "success": "#2ca02c",  # Professional green
    "danger": "#d62728",  # Professional red
    "warning": "#ff7f0e",  # Orange
    "info": "#17a2b8",  # Teal
    "dark": "#343a40",  # Dark gray
    "light": "#f8f9fa",  # Light gray
    "purple": "#9467bd",  # Purple
    "brown": "#8c564b",  # Brown
    "pink": "#e377c2",  # Pink
    "gray": "#7f7f7f",  # Gray
    "olive": "#bcbd22",  # Olive
    "cyan": "#17becf",  # Cyan
}

# Model type colors
model_colors = {
    "CNN": colors["primary"],
    "RNN": colors["secondary"],
    "Proposed": colors["success"],
    "Ablation": colors["purple"],
}

# Model categories
cnn_models = ["VGG16", "MobileNet", "ResNet50", "Xception", "DenseNet121"]
rnn_models = ["LSTM", "Bi-LSTM", "GRU"]
proposed_models = ["Proposed_Model"]
ablation_models = ["Residual_Only", "Inception_Only", "Baseline_CNN"]


def load_and_prepare_data(results_dir="review/apnea_detection_results_20250927_144756"):
    """Load and prepare the apnea detection results data."""
    print("Loading apnea detection results...")

    # Load comprehensive results
    df = pd.read_csv(f"{results_dir}/table6_comprehensive_results.csv")

    # Add model categories
    df["Model_Type"] = df["Model"].apply(
        lambda x: (
            "CNN"
            if x in cnn_models
            else (
                "RNN"
                if x in rnn_models
                else "Proposed" if x in proposed_models else "Ablation"
            )
        )
    )

    # Add model colors
    df["Color"] = df["Model_Type"].map(model_colors)

    # Convert accuracy to percentage
    df["Test_Accuracy_Pct"] = df["Test_Accuracy"] * 100
    df["Train_Accuracy_Pct"] = df["Train_Accuracy"] * 100
    df["Validation_Accuracy_Pct"] = df["Validation_Accuracy"] * 100

    # Calculate efficiency metrics
    df["Parameters_Millions"] = df["Parameters_Trainable"] / 1e6
    df["Efficiency_Score"] = df["Test_Accuracy"] / (df["Parameters_Trainable"] / 1e6)

    # Create performance tiers
    df["Performance_Tier"] = df["Test_Accuracy"].apply(
        lambda x: (
            "Excellent (≥99%)"
            if x >= 0.99
            else (
                "Very Good (95-99%)"
                if x >= 0.95
                else "Good (90-95%)" if x >= 0.90 else "Fair (<90%)"
            )
        )
    )

    print(f"Loaded {len(df)} model results")
    print(f"Model types: {df['Model_Type'].nunique()}")
    print(f"Best performing model: {df.loc[df['Test_Accuracy'].idxmax(), 'Model']}")

    return df


def create_comprehensive_visualization(df):
    """Create comprehensive apnea detection visualization."""
    print("Creating comprehensive apnea detection visualization...")

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))

    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Panel A: Model performance comparison
    ax1 = fig.add_subplot(gs[0, :2])
    create_performance_comparison(df, ax1)

    # Panel B: Model efficiency analysis
    ax2 = fig.add_subplot(gs[0, 2])
    create_efficiency_analysis(df, ax2)

    # Panel C: Ablation study results
    ax3 = fig.add_subplot(gs[1, :])
    create_ablation_analysis(df, ax3)

    # Panel D: Confusion matrix heatmap
    ax4 = fig.add_subplot(gs[2, :2])
    create_confusion_heatmap(df, ax4)

    # Panel E: Model architecture comparison
    ax5 = fig.add_subplot(gs[2, 2])
    create_architecture_comparison(df, ax5)

    # Add overall title
    fig.suptitle(
        "Comprehensive Apnea Detection Model Analysis\nDeep Learning Architecture Comparison",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.savefig(
        "review/apnea_detection_comprehensive.pdf", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("✓ Comprehensive visualization saved as apnea_detection_comprehensive.pdf")


def create_performance_comparison(df, ax):
    """Create model performance comparison chart."""
    # Sort by test accuracy
    df_sorted = df.sort_values("Test_Accuracy", ascending=True)

    # Create horizontal bar plot
    bars = ax.barh(
        df_sorted["Model"],
        df_sorted["Test_Accuracy_Pct"],
        color=df_sorted["Color"],
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, df_sorted["Test_Accuracy_Pct"])):
        width = bar.get_width()
        ax.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2.0,
            f"{acc:.2f}%",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=9,
        )

    ax.set_title("(A) Model Performance Comparison", fontweight="bold", fontsize=12)
    ax.set_xlabel("Test Accuracy (%)", fontsize=11)
    ax.set_xlim(0, 105)
    ax.grid(True, alpha=0.3, axis="x")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=model_colors[model_type], alpha=0.8, label=model_type)
        for model_type in model_colors.keys()
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)


def create_efficiency_analysis(df, ax):
    """Create model efficiency analysis."""
    # Create scatter plot: Parameters vs Accuracy
    for model_type in df["Model_Type"].unique():
        data = df[df["Model_Type"] == model_type]
        ax.scatter(
            data["Parameters_Millions"],
            data["Test_Accuracy_Pct"],
            c=model_colors[model_type],
            s=100,
            alpha=0.8,
            edgecolors="black",
            linewidth=1,
            label=model_type,
        )

    # Add model labels for top performers
    top_models = df.nlargest(5, "Test_Accuracy")
    for _, model in top_models.iterrows():
        ax.annotate(
            model["Model"],
            (model["Parameters_Millions"], model["Test_Accuracy_Pct"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_title("(B) Model Efficiency Analysis", fontweight="bold", fontsize=12)
    ax.set_xlabel("Parameters (Millions)", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.9)


def create_ablation_analysis(df, ax):
    """Create ablation study analysis."""
    # Filter ablation models
    ablation_df = df[df["Model_Type"] == "Ablation"].copy()
    ablation_df = ablation_df.sort_values("Test_Accuracy", ascending=True)

    # Create bar plot
    bars = ax.bar(
        ablation_df["Model"],
        ablation_df["Test_Accuracy_Pct"],
        color=model_colors["Ablation"],
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    # Add value labels
    for bar, acc in zip(bars, ablation_df["Test_Accuracy_Pct"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # Add proposed model for comparison
    proposed_acc = df[df["Model"] == "Proposed_Model"]["Test_Accuracy_Pct"].iloc[0]
    ax.axhline(
        y=proposed_acc,
        color=colors["success"],
        linestyle="--",
        linewidth=2,
        label=f"Proposed Model ({proposed_acc:.2f}%)",
    )

    ax.set_title(
        "(C) Ablation Study: Component Contribution Analysis",
        fontweight="bold",
        fontsize=12,
    )
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(framealpha=0.9)


def create_confusion_heatmap(df, ax):
    """Create confusion matrix heatmap for top models."""
    # Get top 6 models
    top_models = df.nlargest(6, "Test_Accuracy")

    # Create subplot grid for confusion matrices
    n_models = len(top_models)
    cols = 3
    rows = (n_models + cols - 1) // cols

    # Create individual confusion matrix plots
    for idx, (_, model) in enumerate(top_models.iterrows()):
        # Parse confusion matrix
        cm_str = model["Confusion_Matrix"]
        if isinstance(cm_str, str):
            cm = ast.literal_eval(cm_str)
        else:
            cm = cm_str

        # Create subplot
        sub_ax = plt.subplot(rows, cols, idx + 1)

        # Plot confusion matrix
        im = sub_ax.imshow(cm, cmap="Blues", aspect="auto")

        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                sub_ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontweight="bold",
                )

        sub_ax.set_title(
            f"{model['Model']}\n({model['Test_Accuracy_Pct']:.2f}%)",
            fontweight="bold",
            fontsize=10,
        )
        sub_ax.set_xticks([0, 1])
        sub_ax.set_yticks([0, 1])
        sub_ax.set_xticklabels(["Non-Apnea", "Apnea"])
        sub_ax.set_yticklabels(["Non-Apnea", "Apnea"])

        # Add colorbar
        plt.colorbar(im, ax=sub_ax, shrink=0.8)

    # Hide empty subplots
    for idx in range(n_models, rows * cols):
        sub_ax = plt.subplot(rows, cols, idx + 1)
        sub_ax.set_visible(False)

    ax.set_title(
        "(D) Confusion Matrices - Top Performing Models", fontweight="bold", fontsize=12
    )


def create_architecture_comparison(df, ax):
    """Create architecture comparison analysis."""
    # Group by model type and calculate statistics
    arch_stats = (
        df.groupby("Model_Type")
        .agg(
            {
                "Test_Accuracy": ["mean", "std", "count"],
                "Parameters_Trainable": "mean",
                "Test_Loss": "mean",
            }
        )
        .round(4)
    )

    arch_stats.columns = [
        "Mean_Accuracy",
        "Std_Accuracy",
        "Count",
        "Mean_Parameters",
        "Mean_Loss",
    ]

    # Create radar chart data
    categories = ["Mean_Accuracy", "Mean_Parameters", "Mean_Loss"]

    # Normalize data for radar chart
    normalized_data = {}
    for model_type in arch_stats.index:
        values = []
        for cat in categories:
            if cat == "Mean_Accuracy":
                values.append(
                    arch_stats.loc[model_type, cat] * 100
                )  # Convert to percentage
            elif cat == "Mean_Parameters":
                values.append(
                    arch_stats.loc[model_type, cat] / 1e6
                )  # Convert to millions
            else:  # Mean_Loss
                values.append(
                    arch_stats.loc[model_type, cat] * 100
                )  # Scale for visibility
        normalized_data[model_type] = values

    # Create bar plot instead of radar for clarity
    x = np.arange(len(categories))
    width = 0.2

    for i, model_type in enumerate(arch_stats.index):
        values = normalized_data[model_type]
        ax.bar(
            x + i * width,
            values,
            width,
            label=model_type,
            color=model_colors[model_type],
            alpha=0.8,
        )

    ax.set_title("(E) Architecture Comparison", fontweight="bold", fontsize=12)
    ax.set_xlabel("Metrics", fontsize=11)
    ax.set_ylabel("Normalized Values", fontsize=11)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(["Accuracy (%)", "Parameters (M)", "Loss (×100)"])
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)


def create_detailed_performance_analysis(df):
    """Create detailed performance analysis."""
    print("Creating detailed performance analysis...")

    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Panel A: Training vs Test Accuracy
    for model_type in df["Model_Type"].unique():
        data = df[df["Model_Type"] == model_type]
        ax1.scatter(
            data["Train_Accuracy_Pct"],
            data["Test_Accuracy_Pct"],
            c=model_colors[model_type],
            s=100,
            alpha=0.8,
            edgecolors="black",
            linewidth=1,
            label=model_type,
        )

    # Add diagonal line for perfect generalization
    ax1.plot([0, 100], [0, 100], "k--", alpha=0.5, label="Perfect Generalization")

    ax1.set_title("(A) Training vs Test Accuracy", fontweight="bold")
    ax1.set_xlabel("Training Accuracy (%)", fontsize=11)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(framealpha=0.9)

    # Panel B: Loss vs Accuracy
    ax2.scatter(
        df["Test_Loss"],
        df["Test_Accuracy_Pct"],
        c=df["Color"],
        s=100,
        alpha=0.8,
        edgecolors="black",
        linewidth=1,
    )

    # Add model labels for top performers
    top_models = df.nlargest(5, "Test_Accuracy")
    for _, model in top_models.iterrows():
        ax2.annotate(
            model["Model"],
            (model["Test_Loss"], model["Test_Accuracy_Pct"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )

    ax2.set_title("(B) Test Loss vs Test Accuracy", fontweight="bold")
    ax2.set_xlabel("Test Loss", fontsize=11)
    ax2.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Panel C: Precision vs Recall
    # Calculate average precision and recall
    df["Avg_Precision"] = (df["Precision_0"] + df["Precision_1"]) / 2
    df["Avg_Recall"] = (df["Recall_0"] + df["Recall_1"]) / 2

    for model_type in df["Model_Type"].unique():
        data = df[df["Model_Type"] == model_type]
        ax3.scatter(
            data["Avg_Precision"] * 100,
            data["Avg_Recall"] * 100,
            c=model_colors[model_type],
            s=100,
            alpha=0.8,
            edgecolors="black",
            linewidth=1,
            label=model_type,
        )

    ax3.set_title("(C) Precision vs Recall", fontweight="bold")
    ax3.set_xlabel("Average Precision (%)", fontsize=11)
    ax3.set_ylabel("Average Recall (%)", fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.legend(framealpha=0.9)

    # Panel D: Model complexity analysis
    ax4.scatter(
        df["Parameters_Millions"],
        df["Test_Accuracy_Pct"],
        c=df["Color"],
        s=100,
        alpha=0.8,
        edgecolors="black",
        linewidth=1,
    )

    # Add efficiency frontier
    df_sorted = df.sort_values("Parameters_Millions")
    frontier = []
    max_acc = 0
    for _, model in df_sorted.iterrows():
        if model["Test_Accuracy_Pct"] > max_acc:
            frontier.append((model["Parameters_Millions"], model["Test_Accuracy_Pct"]))
            max_acc = model["Test_Accuracy_Pct"]

    if frontier:
        frontier_x, frontier_y = zip(*frontier)
        ax4.plot(
            frontier_x,
            frontier_y,
            "r--",
            alpha=0.7,
            linewidth=2,
            label="Efficiency Frontier",
        )

    ax4.set_title("(D) Model Complexity vs Performance", fontweight="bold")
    ax4.set_xlabel("Parameters (Millions)", fontsize=11)
    ax4.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.legend(framealpha=0.9)

    fig.suptitle(
        "Detailed Performance Analysis", fontsize=16, fontweight="bold", y=0.98
    )
    plt.tight_layout()
    plt.savefig("review/apnea_detailed_performance.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print("✓ Detailed performance analysis saved as apnea_detailed_performance.pdf")


def create_statistical_summary(df):
    """Create statistical summary and analysis."""
    print("Creating statistical summary...")

    # Calculate overall statistics
    total_models = len(df)
    best_accuracy = df["Test_Accuracy"].max()
    best_model = df.loc[df["Test_Accuracy"].idxmax(), "Model"]

    # Performance tiers
    tier_counts = df["Performance_Tier"].value_counts()

    # Model type performance
    type_performance = (
        df.groupby("Model_Type")["Test_Accuracy"].agg(["mean", "std", "count"]).round(4)
    )

    # Create summary visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Panel A: Performance tier distribution
    colors_pie = [
        colors["success"],
        colors["primary"],
        colors["warning"],
        colors["danger"],
    ]
    ax1.pie(
        tier_counts.values,
        labels=tier_counts.index,
        colors=colors_pie[: len(tier_counts)],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax1.set_title("(A) Performance Tier Distribution", fontweight="bold")

    # Panel B: Model type performance
    x = np.arange(len(type_performance.index))
    bars = ax2.bar(
        x,
        type_performance["mean"] * 100,
        color=[model_colors[mt] for mt in type_performance.index],
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    # Add error bars
    ax2.errorbar(
        x,
        type_performance["mean"] * 100,
        yerr=type_performance["std"] * 100,
        fmt="none",
        color="black",
        capsize=5,
    )

    # Add value labels
    for i, (bar, mean, std) in enumerate(
        zip(bars, type_performance["mean"], type_performance["std"])
    ):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{mean*100:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax2.set_title("(B) Performance by Model Type", fontweight="bold")
    ax2.set_ylabel("Mean Test Accuracy (%)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(type_performance.index, rotation=45)
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel C: Accuracy distribution
    ax3.hist(
        df["Test_Accuracy_Pct"],
        bins=15,
        color=colors["primary"],
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )
    ax3.axvline(
        df["Test_Accuracy_Pct"].mean(),
        color=colors["danger"],
        linestyle="--",
        linewidth=2,
        label=f'Mean: {df["Test_Accuracy_Pct"].mean():.2f}%',
    )
    ax3.axvline(
        df["Test_Accuracy_Pct"].median(),
        color=colors["success"],
        linestyle="--",
        linewidth=2,
        label=f'Median: {df["Test_Accuracy_Pct"].median():.2f}%',
    )
    ax3.set_title("(C) Test Accuracy Distribution", fontweight="bold")
    ax3.set_xlabel("Test Accuracy (%)", fontsize=11)
    ax3.set_ylabel("Frequency", fontsize=11)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel D: Parameter efficiency
    ax4.scatter(
        df["Parameters_Millions"],
        df["Efficiency_Score"],
        c=df["Color"],
        s=100,
        alpha=0.8,
        edgecolors="black",
        linewidth=1,
    )

    # Add model labels for most efficient
    top_efficient = df.nlargest(5, "Efficiency_Score")
    for _, model in top_efficient.iterrows():
        ax4.annotate(
            model["Model"],
            (model["Parameters_Millions"], model["Efficiency_Score"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )

    ax4.set_title("(D) Parameter Efficiency Analysis", fontweight="bold")
    ax4.set_xlabel("Parameters (Millions)", fontsize=11)
    ax4.set_ylabel("Efficiency Score (Accuracy/Parameters)", fontsize=11)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(
        "Statistical Summary of Apnea Detection Models",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig("review/apnea_statistical_summary.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print("✓ Statistical summary saved as apnea_statistical_summary.pdf")

    # Print summary statistics
    print(f"\n📊 STATISTICAL SUMMARY:")
    print(f"Total models analyzed: {total_models}")
    print(f"Best performing model: {best_model} ({best_accuracy*100:.2f}%)")
    print(f"Mean test accuracy: {df['Test_Accuracy'].mean()*100:.2f}%")
    print(f"Standard deviation: {df['Test_Accuracy'].std()*100:.2f}%")
    print(f"\nPerformance by model type:")
    for model_type in type_performance.index:
        mean_acc = type_performance.loc[model_type, "mean"] * 100
        std_acc = type_performance.loc[model_type, "std"] * 100
        count = type_performance.loc[model_type, "count"]
        print(f"  {model_type}: {mean_acc:.2f}% ± {std_acc:.2f}% (n={count})")


def main():
    """Main execution function."""
    print("=" * 60)
    print("APNEA DETECTION MODEL VISUALIZATION")
    print("=" * 60)

    # Create output directory
    Path("review").mkdir(exist_ok=True)

    # Load data
    df = load_and_prepare_data()

    # Create visualizations
    create_comprehensive_visualization(df)
    create_detailed_performance_analysis(df)
    create_statistical_summary(df)

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE!")
    print("=" * 60)
    print("Generated files:")
    print("  📊 apnea_detection_comprehensive.pdf - Main comprehensive analysis")
    print("  📊 apnea_detailed_performance.pdf - Detailed performance analysis")
    print("  📊 apnea_statistical_summary.pdf - Statistical summary")
    print("\nAll files saved in the 'review/' directory")


if __name__ == "__main__":
    main()
