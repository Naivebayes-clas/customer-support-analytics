"""
Observational study: Effect of escalation on satisfaction.
No randomization exists in this data → use propensity score matching.
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def load_and_prepare():
    tickets = pd.read_csv("./data/ravenstack_support_tickets.csv")
    accounts = pd.read_csv("./data/ravenstack_accounts.csv")

    # Only tickets with a satisfaction score (drop NaN)
    df = tickets.dropna(subset=["satisfaction_score"]).copy()
    df = df.merge(accounts[["account_id", "industry", "country", "seats", "plan_tier"]],
                  on="account_id", how="left")

    # Treatment: was the ticket escalated?
    df["treatment"] = df["escalation_flag"].astype(int)

    print(f"Tickets with satisfaction scores: {len(df)}")
    print(f"Escalated: {df['treatment'].sum()} ({df['treatment'].mean()*100:.1f}%)")
    return df

def propensity_score_matching(df):
    """Match escalated vs. non-escalated tickets on pre-treatment covariates."""
    # Covariates (all known BEFORE escalation happens)
    cat_cols = ["industry", "country", "plan_tier", "priority"]
    num_cols = ["seats"]

    X = df[cat_cols + num_cols].copy()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    X = X.fillna(0)

    treatment = df["treatment"].values
    outcome = df["satisfaction_score"].values

    # Fit propensity model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_scaled, treatment)
    ps = lr.predict_proba(X_scaled)[:, 1]

    # Nearest-neighbor matching with caliper = 0.2
    caliper = 0.2
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]

    matched_t, matched_c = [], []
    used = set()

    for t in treated_idx:
        dists = np.abs(ps[control_idx] - ps[t])
        for i, c in enumerate(control_idx):
            if c in used:
                continue
            if dists[i] <= caliper:
                matched_t.append(t)
                matched_c.append(c)
                used.add(c)
                break

    matched_t = np.array(matched_t)
    matched_c = np.array(matched_c)
    print(f"\nMatched pairs: {len(matched_t)}")
    print(f"  Treated (escalated): mean satisfaction = {outcome[matched_t].mean():.3f}")
    print(f"  Control (not esc):   mean satisfaction = {outcome[matched_c].mean():.3f}")

    # Test on matched data
    t_stat, p_val = stats.ttest_ind(outcome[matched_t], outcome[matched_c], equal_var=False)
    n1, n2 = len(matched_t), len(matched_c)
    pooled_std = np.sqrt(((n1-1)*outcome[matched_t].std(ddof=1)**2 +
                           (n2-1)*outcome[matched_c].std(ddof=1)**2) / (n1+n2-2))
    cohens_d = (outcome[matched_t].mean() - outcome[matched_c].mean()) / pooled_std

    print(f"\nResults (matched):")
    print(f"  t = {t_stat:.3f}")
    print(f"  p = {p_val:.4f}")
    print(f"  Cohen's d = {cohens_d:.4f}")
    print(f"  95% CI: [{(outcome[matched_t].mean()-outcome[matched_c].mean()) - 1.96*pooled_std*np.sqrt(1/n1+1/n2):.3f}, "
          f"{(outcome[matched_t].mean()-outcome[matched_c].mean()) + 1.96*pooled_std*np.sqrt(1/n1+1/n2):.3f}]")

    return matched_t, matched_c

if __name__ == "__main__":
    df = load_and_prepare()
    propensity_score_matching(df)   
