"""
Customer segmentation: K-Means on aggregated customer-level features.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os

os.makedirs("./output/figures", exist_ok=True)

def build_features():
    """Aggregate to customer level."""
    accounts = pd.read_csv("./data/ravenstack_accounts.csv")
    subs = pd.read_csv("./data/ravenstack_subscriptions.csv")
    tickets = pd.read_csv("./data/ravenstack_support_tickets.csv")
    usage = pd.read_csv("./data/ravenstack_feature_usage.csv")

    # Subscription features
    sub_agg = subs.groupby("account_id").agg(
        mrr=("mrr_amount", "mean"),
        arr=("arr_amount", "mean"),
        n_subscriptions=("subscription_id", "count"),
        seats=("seats", "max"),
    ).reset_index()

    # Ticket features
    ticket_agg = tickets.groupby("account_id").agg(
        n_tickets=("ticket_id", "count"),
        mean_resolution_time=("resolution_time_hours", "mean"),
        mean_satisfaction=("satisfaction_score", "mean"),
        n_escalations=("escalation_flag", "sum"),
    ).reset_index()

    # Usage features
    sub_map = subs[["subscription_id", "account_id"]].drop_duplicates()
    usage = usage.merge(sub_map, on="subscription_id")
    usage_agg = usage.groupby("account_id").agg(
        total_usage=("usage_count", "sum"),
        total_errors=("error_count", "sum"),
    ).reset_index()

    # Merge all
    df = accounts.merge(sub_agg, on="account_id", how="left")
    df = df.merge(ticket_agg, on="account_id", how="left")
    df = df.merge(usage_agg, on="account_id", how="left")

    # Fill missing
    fill_cols = ["mrr", "arr", "n_subscriptions", "n_tickets",
             "mean_resolution_time", "mean_satisfaction", "n_escalations",
             "total_usage", "total_errors"]
    df[fill_cols] = df[fill_cols].fillna(0)

    return df

def segment(df, n_clusters=4):
    """K-Means + silhouette."""
    
    feature_cols = ["mrr", "n_tickets", "mean_resolution_time",
                "mean_satisfaction", "total_usage", "total_errors"]
                
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)

    print(f"Silhouette score: {sil:.4f}")

    # Cluster profiles
    df["cluster"] = labels
    profile = df.groupby("cluster")[feature_cols].mean()
    profile["n_customers"] = df.groupby("cluster").size()
    print("\nCluster Profiles:")
    print(profile.round(2))

    # 2D scatter (first 2 PCA components for visualization)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="viridis", alpha=0.4, s=15)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")
    plt.title(f"Customer Segments (K={n_clusters}, silhouette={sil:.3f})")
    plt.savefig("./output/figures/segments.png", dpi=150, bbox_inches="tight")
    plt.show()

    df.to_csv("./output/segments.csv", index=False)
    return labels, sil

if __name__ == "__main__":
    df = build_features()
    labels, sil = segment(df)   
