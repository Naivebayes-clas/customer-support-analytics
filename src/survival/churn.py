"""
Churn survival analysis on RavenStack data.
Duration: signup_date → churn_date (or censoring date)
Event: 1 = churned, 0 = censored
"""
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import os

os.makedirs("./output/figures", exist_ok=True)

def load_data():
    accounts = pd.read_csv("./data/ravenstack_accounts.csv", parse_dates=["signup_date"])
    churn = pd.read_csv("./data/ravenstack_churn_events.csv", parse_dates=["churn_date"])
    subs = pd.read_csv("./data/ravenstack_subscriptions.csv", parse_dates=["start_date", "end_date"])
    tickets = pd.read_csv("./data/ravenstack_support_tickets.csv", parse_dates=["submitted_at", "closed_at"])
    usage = pd.read_csv("./data/ravenstack_feature_usage.csv", parse_dates=["usage_date"])

    return accounts, churn, subs, tickets, usage

def build_survival_df(accounts, churn, subs, tickets, usage):
    """
    Build a customer-level DataFrame with:
    - duration: days from signup to churn (or censoring)
    - event: 1 = churned, 0 = censored
    - covariates: aggregated from subs, tickets, usage
    """
    # Merge churn events (left join — accounts without churn get NaN)
    df = accounts.merge(churn[["account_id", "churn_date", "reason_code", "refund_amount_usd"]],
                        on="account_id", how="left")
                        
    # DEDUPLICATE: keep only the FIRST churn event per account
    df = df.sort_values("churn_date", na_position="last").drop_duplicates(
        subset="account_id", keep="first"
    ).reset_index(drop=True)
    
    # Censoring date = max churn_date in the dataset (standard practice)
    max_date = df["churn_date"].max()
    df["churn_date"] = df["churn_date"].fillna(max_date)

    # Duration in days
    df["duration_days"] = (df["churn_date"] - df["signup_date"]).dt.days
    df["duration_days"] = df["duration_days"].clip(lower=1)  # avoid 0/negative

    # Event: 1 = actually churned, 0 = censored
    df["event"] = df["churn_flag"].astype(int)

    # --- Covariates from subscriptions (first subscription per account) ---
    sub_agg = subs.groupby("account_id").agg(
        mrr=("mrr_amount", "first"),
        plan_enterprise=("plan_tier", lambda x: (x == "Enterprise").sum() > 0),
        plan_pro=("plan_tier", lambda x: (x == "Pro").sum() > 0),
        n_subscriptions=("subscription_id", "count"),
    ).reset_index()
    df = df.merge(sub_agg, on="account_id", how="left")

    # --- Covariates from support tickets ---
    ticket_agg = tickets.groupby("account_id").agg(
        n_tickets=("ticket_id", "count"),
        mean_resolution_time=("resolution_time_hours", "mean"),
        n_escalations=("escalation_flag", "sum"),
        mean_satisfaction=("satisfaction_score", "mean"),
    ).reset_index()
    df = df.merge(ticket_agg, on="account_id", how="left")

    # --- Covariates from feature usage ---
    usage_agg = usage.groupby("subscription_id").agg(
        total_usage_count=("usage_count", "sum"),
        total_error_count=("error_count", "sum"),
        n_beta_features=("is_beta_feature", lambda x: x.sum() > 0),
    ).reset_index()
    # Map subscription_id → account_id
    sub_map = subs[["subscription_id", "account_id"]].drop_duplicates()
    usage_agg = usage_agg.merge(sub_map, on="subscription_id")
    usage_agg = usage_agg.groupby("account_id").agg(
        total_usage_count=("total_usage_count", "sum"),
        total_error_count=("total_error_count", "sum"),
        n_beta_features=("n_beta_features", "max"),
    ).reset_index()
    df = df.merge(usage_agg, on="account_id", how="left")

    # Fill missing (accounts with no tickets/usage)
    df[["n_tickets", "mean_resolution_time", "n_escalations", "mean_satisfaction",
        "total_usage_count", "total_error_count", "n_beta_features"]] = \
        df[["n_tickets", "mean_resolution_time", "n_escalations", "mean_satisfaction",
            "total_usage_count", "total_error_count", "n_beta_features"]].fillna(0)

    df["mrr"] = df["mrr"].fillna(df["mrr"].median())

    # Drop reason_code (post-treatment — only known after churn)
    df = df.drop(columns=["churn_date", "reason_code", "refund_amount_usd", "account_name", "churn_flag"])

    print(f"Survival DF shape: {df.shape}")
    print(f"Churn rate: {df['event'].mean():.3f}")
    print(f"Median duration: {df['duration_days'].median():.0f} days")
    return df

def fit_cox(df):
    """Fit Cox PH model with pre-treatment covariates only."""
    covariates = ["mrr", "plan_enterprise", "plan_pro", "n_tickets",
                  "mean_resolution_time", "n_escalations", "total_usage_count",
                  "total_error_count", "n_beta_features"]

    # Drop rows with NaN in covariates
    cph_df = df[["duration_days", "event"] + covariates].dropna()
    print(f"Cox model: {len(cph_df)} rows, {cph_df['event'].sum()} events")

    cph = CoxPHFitter()
    cph.fit(cph_df, duration_col="duration_days", event_col="event")
    summary = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]   
    print("\nCox PH Results:")
    print(summary)
     # PH assumption check (Schoenfeld residuals)
    print("\nPH Assumption Check (Schoenfeld residuals):")
    try:
        ph = cph.check_assumptions(cph_df)
        print(ph)
    except Exception as e:
        print(f"  (PH check failed: {e})")
        
    return cph

def plot_km(df, group_col="plan_tier"):
    """Kaplan-Meier curves by plan tier + log-rank test."""
    # Need plan_tier in df — re-merge from accounts

    groups = df["plan_tier"].unique()
    plt.figure(figsize=(10, 6))

    kmfs = []
    for g in sorted(groups):
        subset = df[df["plan_tier"] == g]
        kmf = KaplanMeierFitter()
        kmf.fit(subset["duration_days"], subset["event"], label=g)
        kmf.plot_survival_function()
        kmfs.append((g, subset))

    # Log-rank: compare Basic vs Enterprise
    basic = df[df["plan_tier"] == "Basic"]
    ent = df[df["plan_tier"] == "Enterprise"]
    if len(basic) > 0 and len(ent) > 0:
        lr = logrank_test(basic["duration_days"], ent["duration_days"],
                          event_observed_A=basic["event"], event_observed_B=ent["event"])
        print(f"\nLog-rank (Basic vs Enterprise): p = {lr.p_value:.4f}")
        plt.title(f"Churn Survival by Plan Tier (log-rank p={lr.p_value:.3f})")

    plt.xlabel("Days Since Signup")
    plt.ylabel("P(Still Active)")
    plt.legend()
    plt.savefig("./output/figures/km_churn_by_plan.png", dpi=150, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    accounts, churn, subs, tickets, usage = load_data()
    df = build_survival_df(accounts, churn, subs, tickets, usage)
    cph = fit_cox(df)
    plot_km(df)   
