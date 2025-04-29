import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='latin1')


def basic_analysis(df):
    summary = {
        "columns": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "description": df.describe(include='all').to_dict()
    }
    return summary


def create_charts(df, output_folder):
    charts = []
    numeric_cols = df.select_dtypes(include='number').columns

    if len(numeric_cols) > 0:
        plt.figure()
        sns.histplot(df[numeric_cols[0]].dropna(), kde=True)
        plt.title(f'Distribution of {numeric_cols[0]}')
        path1 = f"{output_folder}/histogram.png"
        plt.savefig(path1)
        charts.append(path1)
        plt.close()

    if len(numeric_cols) > 1:
        plt.figure()
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        path2 = f"{output_folder}/correlation.png"
        plt.savefig(path2)
        charts.append(path2)
        plt.close()

    return charts


def save_readme(summary, chart_paths, output_folder):
    md = "# Automated Data Analysis Report\n\n"
    md += "## Dataset Summary\n"
    md += f"**Columns**: {summary['columns']}\n\n"
    md += f"**Data Types**: {summary['data_types']}\n\n"
    md += f"**Missing Values**: {summary['missing_values']}\n\n"

    md += "\n## Key Insights\n"
    md += "- This is a simple automated analysis.\n"
    md += "- Charts and summary statistics were generated.\n"
    md += "- GPT-based storytelling skipped (offline mode).\n"

    md += "\n## Visualizations\n"
    for chart in chart_paths:
        md += f"![Chart]({os.path.basename(chart)})\n"

    with open(f"{output_folder}/README.md", "w", encoding="utf-8") as f:
        f.write(md)


def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py dataset.csv")
        return

    file = sys.argv[1]
    folder = os.path.splitext(os.path.basename(file))[0]
    os.makedirs(folder, exist_ok=True)

    df = load_data(file)
    summary = basic_analysis(df)
    charts = create_charts(df, folder)
    save_readme(summary, charts, folder)

    print(f"✅ Done. Check the folder '{folder}' for output.")


if __name__ == "__main__":
    main()
