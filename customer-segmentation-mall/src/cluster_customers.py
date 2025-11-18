import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # goes from src/ to project root
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Mall_Customers.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def preprocess(df):
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, df

def run_kmeans(X_scaled, k=6):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return labels

def build_cluster_profile(df):
    cluster_profile = df.groupby('cluster').agg(
        count=('CustomerID', 'count'),
        avg_age=('Age', 'mean'),
        avg_income=('Annual Income (k$)', 'mean'),
        avg_spend=('Spending Score (1-100)', 'mean')
    ).reset_index()
    return cluster_profile

def generate_report(df, cluster_profile, k):
    lines = []

    lines.append("# Customer Segmentation Report\n\n")
    lines.append("## 1. Introduction\n")
    lines.append(
        "This report presents a customer segmentation analysis using the Mall Customers dataset. "
        "The objective is to identify distinct customer groups based on age, income, and spending behaviour, "
        "and to derive actionable insights for targeted marketing.\n\n"
    )

    lines.append("## 2. Data Overview\n")
    lines.append(f"- Number of customers: {len(df)}\n")
    lines.append("- Features used: Age, Annual Income (k$), Spending Score (1-100)\n\n")

    lines.append("## 3. Methodology\n")
    lines.append(
        "- Preprocessing: Selected numeric features and applied standardization.\n"
        f"- Clustering algorithm: K-means with k = {k} clusters.\n"
        "- Evaluation: Number of clusters chosen based on elbow and silhouette methods (see notebook for details).\n\n"
    )

    lines.append("## 4. Cluster Profiles\n")

    for _, row in cluster_profile.iterrows():
        c = int(row['cluster'])
        lines.append(f"### Cluster {c}\n")
        lines.append(f"- Number of customers: {row['count']}\n")
        lines.append(f"- Average age: {row['avg_age']:.1f}\n")
        lines.append(f"- Average annual income (k$): {row['avg_income']:.1f}\n")
        lines.append(f"- Average spending score (1-100): {row['avg_spend']:.1f}\n")
        lines.append("- Description: _Add a human-friendly label here (e.g., 'Young high spenders')._\n\n")

    lines.append("## 5. Actionable Insights\n")
    lines.append(
        "Below are example types of strategies the mall could adopt for different clusters. "
        "You should refine these based on the exact profiles observed:\n\n"
    )
    lines.append("- High-income, low-spending clusters: focus on premium experiences and personalized offers.\n")
    lines.append("- Young, high-spending clusters: emphasize fashion, entertainment, and digital campaigns.\n")
    lines.append("- Budget-conscious clusters: promote discounts, bundles, and loyalty points.\n\n")

    lines.append("## 6. Conclusion\n")
    lines.append(
        "The analysis highlights distinct segments within the mall's customer base. "
        "By tailoring marketing campaigns and offerings to each segment, the mall can increase engagement, "
        "optimize promotional budgets, and improve overall customer satisfaction.\n"
    )

    report_path = REPORTS_DIR / "customer_segmentation.md"
    with open(report_path, "w") as f:
        f.writelines(lines)

    print(f"Report written to {report_path}")

def main():
    k = 6  # set this to the value you chose in the notebook
    df = load_data()
    X_scaled, df = preprocess(df)
    labels = run_kmeans(X_scaled, k=k)
    df['cluster'] = labels

    cluster_profile = build_cluster_profile(df)
    print(cluster_profile)

    generate_report(df, cluster_profile, k)

if __name__ == "__main__":
    main()
