"""
Observational study: Effect of escalation on satisfaction.
Uses the PropensityScoreMatcher and ABTester from tester.py.
"""
import pandas as pd
import numpy as np
from src.ab_testing.tester import ABTester, PropensityScoreMatcher

def load_and_prepare():
    tickets = pd.read_csv("./data/ravenstack_support_tickets.csv")
    accounts = pd.read_csv("./data/ravenstack_accounts.csv")

    df = tickets.dropna(subset=["satisfaction_score"]).copy()
    df = df.merge(accounts[["account_id", "industry", "country", "seats", "plan_tier"]],
                  on="account_id", how="left")
    df["treatment"] = df["escalation_flag"].astype(int)

    print(f"Tickets with satisfaction scores: {len(df)}")
    print(f"Escalated: {df['treatment'].sum()} ({df['treatment'].mean()*100:.1f}%)")
    return df

def main():
    df = load_and_prepare()

    # Covariates (pre-treatment only)
    cat_cols = ["industry", "country", "plan_tier", "priority"]
    num_cols = ["seats"]
    X = df[cat_cols + num_cols].copy()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True).fillna(0)

    treatment = df["treatment"].values
    outcome = df["satisfaction_score"].values

    # PSM (from tester.py)
    psm = PropensityScoreMatcher(caliper=0.2)
    psm.fit(X, treatment)
    matched_t, matched_c = psm.match(treatment)

    print(f"\nMatched pairs: {len(matched_t)}")

    # Welch's t-test (from tester.py)
    result = ABTester.welch_t_test(outcome[matched_t], outcome[matched_c])

    print(f"  Treated mean: {outcome[matched_t].mean():.3f}")
    print(f"  Control mean: {outcome[matched_c].mean():.3f}")
    print(f"  t = {result['t']:.3f}")
    print(f"  p = {result['p_value']:.4f}")
    print(f"  Cohen's d = {result['cohens_d']:.4f}")

if __name__ == "__main__":
    main()   
