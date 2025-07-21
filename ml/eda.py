import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
bivariate_dir = os.path.join("eda_plots", "bivariate")
os.makedirs(bivariate_dir, exist_ok=True)

def generate_bivariate_plots(df, target_col='Converted'):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=target_col, y=col, data=df)
        plt.title(f'{col} vs. {target_col}')
        plt.xlabel(target_col)
        plt.ylabel(col)
        save_path = os.path.join(bivariate_dir, f'bivariate_{col}_vs_{target_col}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved boxplot: {save_path}")
        print(f"📊 Mean {col} by {target_col}:\n{df.groupby(target_col)[col].mean()}\n")

    for col in categorical_cols:
        plt.figure(figsize=(12, 7))
        sns.countplot(x=col, hue=target_col, data=df)
        plt.title(f'{col} vs. {target_col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=target_col)
        plt.tight_layout()
        save_path = os.path.join(bivariate_dir, f'bivariate_{col}_vs_{target_col}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved countplot: {save_path}")
        print(f"📊 Cross-tabulation of {col} and {target_col}:\n{pd.crosstab(df[col], df[target_col])}\n")
        print(f"📊 Proportions within each {col} category:\n{pd.crosstab(df[col], df[target_col], normalize='index')}\n")

    # Correlation Heatmap
    plt.figure(figsize=(14, 10))
    corr = df[numerical_cols + [target_col]].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    heatmap_path = os.path.join(bivariate_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved correlation heatmap: {heatmap_path}")


def generate_univariate_plots(df):
    plot_dir = "eda_plots/univariate_analysis"
    os.makedirs(plot_dir, exist_ok=True)

    numerical_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include='object').columns

    # 🔢 Univariate plots for numerical columns
    for col in numerical_cols:
        # Histogram + KDE
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        hist_path = os.path.join(plot_dir, f'univariate_hist_{col}.png')
        plt.savefig(hist_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved histogram: {hist_path}")

        # Boxplot
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df[col], color='lightcoral')
        plt.title(f'Box plot of {col}')
        plt.xlabel(col)
        box_path = os.path.join(plot_dir, f'univariate_box_{col}.png')
        plt.savefig(box_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved boxplot: {box_path}")

        print(f"📊 Descriptive Statistics for {col}:\n{df[col].describe()}\n")

    # 🏷️ Univariate plots for categorical columns
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df[col], order=df[col].value_counts().index, palette='viridis')
        plt.title(f'Count plot of {col}')
        plt.xlabel('Count')
        plt.ylabel(col)
        countplot_path = os.path.join(plot_dir, f'univariate_count_{col}.png')
        plt.savefig(countplot_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved countplot: {countplot_path}")

        print(f"📊 Value Counts for {col}:\n{df[col].value_counts()}\n")
        print(f"📊 Proportions for {col}:\n{df[col].value_counts(normalize=True)}\n")


df = pd.read_csv("Lead Scoring.csv")
generate_bivariate_plots(df, target_col='Converted')
generate_univariate_plots(df)
