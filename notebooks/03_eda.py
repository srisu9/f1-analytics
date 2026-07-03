import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set aesthetic styling
plt.style.use('dark_background')
sns.set_theme(style="whitegrid", rc={
    "axes.facecolor": "#1a1a1a",
    "figure.facecolor": "#121212",
    "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#b0b0b0",
    "ytick.color": "#b0b0b0",
    "grid.color": "#2d2d2d",
    "axes.edgecolor": "#2d2d2d"
})
F1_RED = "#e10600"
F1_DARK_RED = "#900000"
ACCENT_BLUE = "#00b2ff"
CHARCOAL = "#222222"

def load_data():
    csv_path = "data/processed/model_ready.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {csv_path}. Please run data cleaning first.")
    return pd.read_csv(csv_path)

def generate_eda():
    df = load_data()
    fig_dir = "reports/figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    print("Generating EDA Plots...")
    
    # 1. Target Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    counts = df["Top10"].value_counts()
    
    # Bar plot
    sns.barplot(x=counts.index, y=counts.values, ax=axes[0], palette=[CHARCOAL, F1_RED], hue=counts.index, legend=False)
    axes[0].set_title("Top 10 Finish Count (Target Variable)", fontsize=14, fontweight="bold", pad=15)
    axes[0].set_xlabel("Finished in Top 10", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["No (Pos > 10)", "Yes (Pos <= 10)"])
    
    # Pie plot
    axes[1].pie(counts, labels=["No (Pos > 10)", "Yes (Pos <= 10)"], autopct='%1.1f%%', 
                colors=[CHARCOAL, F1_RED], startangle=90, textprops={'fontsize': 12, 'color': 'white'},
                wedgeprops={'edgecolor': '#121212', 'linewidth': 2})
    axes[1].set_title("Top 10 Finish Proportion", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/01_target_distribution.png", dpi=150, facecolor='#121212')
    plt.close()
    
    # 2. Grid Position vs Top10
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Top10", y="grid", palette=[CHARCOAL, F1_RED], hue="Top10", legend=False)
    plt.title("Grid Position vs. Top 10 Finish", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Finished in Top 10", fontsize=12)
    plt.ylabel("Starting Grid Position", fontsize=12)
    plt.xticks([0, 1], ["No", "Yes"])
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/02_grid_vs_top10.png", dpi=150, facecolor='#121212')
    plt.close()
    
    # 3. Qualifying Position vs Top10
    plt.figure(figsize=(10, 6))
    # Drop rows where qualifying position is NaN for visualization
    quali_df = df.dropna(subset=["qualifying_position"])
    sns.violinplot(data=quali_df, x="Top10", y="qualifying_position", palette=[CHARCOAL, F1_RED], hue="Top10", legend=False, split=False)
    plt.title("Qualifying Position vs. Top 10 Finish", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Finished in Top 10", fontsize=12)
    plt.ylabel("Qualifying Position", fontsize=12)
    plt.xticks([0, 1], ["No", "Yes"])
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/03_qualifying_vs_top10.png", dpi=150, facecolor='#121212')
    plt.close()
    
    # 4. Constructor Performance
    plt.figure(figsize=(12, 8))
    # Filter to constructors with a reasonable number of races (e.g. > 100 entries)
    top_constructors = df["constructor_name"].value_counts()[df["constructor_name"].value_counts() > 100].index
    const_perf = df[df["constructor_name"].isin(top_constructors)].groupby("constructor_name")["Top10"].mean().sort_values(ascending=False).head(20)
    
    sns.barplot(x=const_perf.values, y=const_perf.index, palette="Reds_r", hue=const_perf.index, legend=False)
    plt.title("Top 20 Constructors by Top 10 Finish Rate (Min 100 Entries)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Top 10 Finish Rate", fontsize=12)
    plt.ylabel("Constructor Name", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/04_constructor_performance.png", dpi=150, facecolor='#121212')
    plt.close()
    
    # 5. Constructor Nationality
    plt.figure(figsize=(12, 6))
    top_nations = df["constructor_nationality"].value_counts().head(10).index
    sns.countplot(data=df[df["constructor_nationality"].isin(top_nations)], 
                  x="constructor_nationality", order=top_nations, palette="viridis", hue="constructor_nationality", legend=False)
    plt.title("Top 10 Constructor Nationalities by Entry Count", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Nationality", fontsize=12)
    plt.ylabel("Number of Entries", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/05_constructor_nationality.png", dpi=150, facecolor='#121212')
    plt.close()
    
    # 6. Year-wise Trends
    plt.figure(figsize=(12, 6))
    yearly_grid_size = df.groupby("year")["grid"].max()
    yearly_top10_rate = df.groupby("year")["Top10"].mean() * 100
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    sns.lineplot(x=yearly_top10_rate.index, y=yearly_top10_rate.values, ax=ax1, color=F1_RED, linewidth=2.5, label="Top 10 Rate (%)")
    sns.lineplot(x=yearly_grid_size.index, y=yearly_grid_size.values, ax=ax2, color=ACCENT_BLUE, linewidth=2, linestyle="--", label="Max Grid Size")
    
    ax1.set_title("Historical Trends: Top 10 Finish Rate vs. Max Grid Size", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Top 10 Finish Rate (%)", color=F1_RED, fontsize=12)
    ax2.set_ylabel("Max Grid Size (Starting Cars)", color=ACCENT_BLUE, fontsize=12)
    
    ax1.tick_params(axis='y', labelcolor=F1_RED)
    ax2.tick_params(axis='y', labelcolor=ACCENT_BLUE)
    
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/06_year_wise_trends.png", dpi=150, facecolor='#121212')
    plt.close()
    
    # 7. Circuit Analysis
    plt.figure(figsize=(12, 8))
    top_circuits = df["circuit_name"].value_counts().head(20).index
    circuit_perf = df[df["circuit_name"].isin(top_circuits)].groupby("circuit_name")["Top10"].mean().sort_values(ascending=False)
    
    sns.barplot(x=circuit_perf.values, y=circuit_perf.index, palette="Blues_r", hue=circuit_perf.index, legend=False)
    plt.title("Top 20 Most Frequent Circuits by Top 10 Finish Rate", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Top 10 Finish Rate", fontsize=12)
    plt.ylabel("Circuit Name", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/07_circuit_performance.png", dpi=150, facecolor='#121212')
    plt.close()
    
    # 8. Correlation Matrix
    plt.figure(figsize=(10, 8))
    numerical_cols = ["grid", "qualifying_position", "year", "round", "lat", "lng", "alt", "Top10"]
    corr = df[numerical_cols].corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1,
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title("Correlation Matrix of Numerical Features", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/08_correlation_matrix.png", dpi=150, facecolor='#121212')
    plt.close()
    
    # 9. Missing Values Bar Chart
    plt.figure(figsize=(12, 6))
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    
    if len(missing_pct) > 0:
        sns.barplot(x=missing_pct.values, y=missing_pct.index, palette="magma", hue=missing_pct.index, legend=False)
        plt.title("Percentage of Missing Values per Column", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Percentage Missing (%)", fontsize=12)
        plt.ylabel("Feature Column", fontsize=12)
    else:
        plt.text(0.5, 0.5, "No Missing Values Found!", fontsize=14, ha="center", va="center")
        plt.title("Missing Values (None Found)", fontsize=14, fontweight="bold", pad=15)
        
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/09_missing_values.png", dpi=150, facecolor='#121212')
    plt.close()
    
    # 10. Grid Position Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x="grid", kde=True, bins=df["grid"].nunique(), color=F1_RED, edgecolor="#121212", linewidth=1.5)
    plt.title("Distribution of Starting Grid Positions", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Starting Grid Position", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/10_grid_distribution.png", dpi=150, facecolor='#121212')
    plt.close()
    
    print("All 10 EDA Plots generated and saved successfully to reports/figures/")

if __name__ == "__main__":
    generate_eda()
