"""
Simulation study: Effect of AI-assisted support on churn.
Uses real RavenStack account features; simulates treatment + outcome.

Run: python -m src.survival.ai_simulation
"""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import os

np.random.seed(42)
OUT = "./output"
os.makedirs(f"{OUT}/figures", exist_ok=True)

# ─── 1. Load REAL account features ───────────────────────────────────────────
accounts = pd.read_csv("./data/ravenstack_accounts.csv", parse_dates=["signup_date"])
subs = pd.read_csv("./data/ravenstack_subscriptions.csv")
tickets = pd.read_csv("./data/ravenstack_support_tickets.csv")

# Real covariates (these are OBSERVED, not simulated)
sub_agg = subs.groupby("account_id").agg(
    mrr=("mrr_amount", "first"),
    plan_enterprise=("plan_tier", lambda x: (x == "Enterprise").sum() > 0),
).reset_index()

ticket_agg = tickets.groupby("account_id").agg(
    n_tickets=("ticket_id", "count"),
    mean_resolution_time=("resolution_time_hours", "mean"),
).reset_index()

df = accounts.merge(sub_agg, on="account_id", how="left")
df = df.merge(ticket_agg, on="account_id", how="left")
df[["n_tickets", "mean_resolution_time"]] = df[["n_tickets", "mean_resolution_time"]].fillna(0)
df["mrr"] = df["mrr"].fillna(df["mrr"].median())

print(f"Real accounts: {len(df)}")
print(f"Real covariates: mrr, plan_enterprise, n_tickets, mean_resolution_time")

# ─── 2. Simulate treatment assignment (randomized 50/50) ────────────────────
df["ai_assisted"] = np.random.binomial(1, 0.5, size=len(df))
print(f"\nTreatment: AI-assisted = {df['ai_assisted'].sum()}, Control = {(1-df['ai_assisted']).sum()}")

# ─── 3. Simulate the MECHANISM (what AI actually changes) ───────────────────
# AI-assisted accounts get:
#   - 15% faster resolution time
#   - 0.2 higher CSAT (on 5-point scale)
# These are the "intermediate outcomes" that would causally affect churn.

df["sim_resolution_time"] = df["mean_resolution_time"] * np.where(
    df["ai_assisted"] == 1, 0.85, 1.0  # 15% faster if AI-assisted
)
df["sim_csat"] = np.clip(
    4.0 + np.random.normal(0, 0.3, len(df)) + np.where(df["ai_assisted"] == 1, 0.2, 0),
    1, 5
)

# ─── 4. Simulate the OUTCOME (churn) ─────────────────────────────────────────
# Base hazard: ~22% annual churn (matches real RavenStack rate)
# AI effect: 20% hazard reduction (the "true" effect we want to detect)

# Simulate time-to-churn using exponential model (for simplicity)
# Real median duration was ~124 days; use that as scale
real_median_days = 124
base_rate = np.log(2) / real_median_days  # ~0.56% per day

# AI-assisted accounts get 20% lower hazard
hazard = np.where(df["ai_assisted"] == 1, base_rate * 0.80, base_rate)

# Add covariate effects (real features influence churn)
# Higher MRR → slightly lower churn (larger accounts stickier)
mrr_effect = 1 - 0.00001 * (df["mrr"] - df["mrr"].mean())
mrr_effect = np.clip(mrr_effect, 0.7, 1.3)

# Enterprise → slightly higher churn (matches your Cox finding direction)
plan_effect = np.where(df["plan_enterprise"], 1.3, 1.0)

# More tickets → slightly higher churn (support burden)
ticket_effect = 1 + 0.02 * (df["n_tickets"] - df["n_tickets"].mean())
ticket_effect = np.clip(ticket_effect, 0.8, 1.5)

# Combined hazard
final_hazard = hazard * mrr_effect * plan_effect * ticket_effect

# Simulate churn times (exponential)
df["sim_duration_days"] = np.random.exponential(1 / final_hazard)
df["sim_duration_days"] = df["sim_duration_days"].clip(1, 730).astype(int)

# Censoring: some accounts haven't churned yet (observed for < 365 days)
max_observed = 365
df["sim_event"] = (df["sim_duration_days"] <= max_observed).astype(int)
df["sim_duration_days"] = df["sim_duration_days"].clip(upper=max_observed)

print(f"\nSimulated churn rate: {df['sim_event'].mean():.3f}")
print(f"  AI-assisted churn: {df[df['ai_assisted']==1]['sim_event'].mean():.3f}")
print(f"  Control churn:     {df[df['ai_assisted']==0]['sim_event'].mean():.3f}")

# ─── 5. Fit Cox PH on simulated data ─────────────────────────────────────────
cox_df = df[["sim_duration_days", "sim_event", "ai_assisted", "mrr",
             "plan_enterprise", "n_tickets"]].copy()

cph = CoxPHFitter()
cph.fit(cox_df, duration_col="sim_duration_days", event_col="sim_event")
summary = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
print("\nCox PH (simulated):")
print(summary)

# ─── 6. Kaplan-Meier + log-rank ─────────────────────────────────────────────
ai = df[df["ai_assisted"] == 1]
ctrl = df[df["ai_assisted"] == 0]

kmf_ai = KaplanMeierFitter().fit(ai["sim_duration_days"], ai["sim_event"], label="AI-Assisted")
kmf_ctrl = KaplanMeierFitter().fit(ctrl["sim_duration_days"], ctrl["sim_event"], label="Control")

lr = logrank_test(ai["sim_duration_days"], ctrl["sim_duration_days"],
                  event_observed_A=ai["sim_event"], event_observed_B=ctrl["sim_event"])
print(f"\nLog-rank (AI vs Control): p = {lr.p_value:.6f}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
kmf_ai.plot_survival_function(ax=ax)
kmf_ctrl.plot_survival_function(ax=ax)
ax.set_title(f"Simulated Churn Survival: AI-Assisted vs. Control\n(log-rank p = {lr.p_value:.4f})")
ax.set_xlabel("Days Since Signup")
ax.set_ylabel("P(Still Active)")
ax.legend()
plt.savefig(f"{OUT}/figures/ai_churn_simulation.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ {OUT}/figures/ai_churn_simulation.png")

# ─── 7. Power analysis: how many accounts needed? ───────────────────────────
from scipy.stats import norm

print("\n─── Power Analysis ───")
print("True effect: 20% hazard reduction (HR = 0.80)")
print("Base annual churn: 22%")
print()

for n_total in [500, 1000, 2000, 5000, 10000]:
    n_per_group = n_total // 2
    events_per_group = n_per_group * 0.22  # expected events
    # Power for Cox with binary treatment
    # Approximate: power = Φ(√events × |log(HR)| / √2 − z_α/2)
    log_hr = abs(np.log(0.80))
    z_alpha = norm.ppf(0.975)
    power = norm.cdf(np.sqrt(events_per_group) * log_hr / np.sqrt(2) - z_alpha)
    print(f"  n = {n_total:>5} ({n_per_group}/group, ~{int(events_per_group)} events): power = {power:.3f}")

print(f"\n✓ Done. True HR = 0.80; detected HR = {summary.loc['ai_assisted', 'exp(coef)']:.3f}")   
