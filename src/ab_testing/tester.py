"""
A/B Testing & Causal Inference Framework
- Power analysis (sample size, MDE)
- Hypothesis tests (t, Welch, z, log-rank, Cox PH)
- Propensity score matching (observational studies)
- Sequential monitoring (Bonferroni)
"""

import numpy as np
from scipy import stats
from scipy.stats import norm
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd


class PowerAnalyzer:
    """Pre-study and post-hoc power calculations."""

    @staticmethod
    def sample_size_two_proportions(p1: float, p2: float, alpha: float = 0.05,
                                     power: float = 0.80, two_sided: bool = True) -> int:
        """
        Sample size per group for two-proportion z-test.

        Args:
            p1: expected proportion in control
            p2: expected proportion in treatment
            alpha: significance level
            power: desired power (1 - beta)
            two_sided: two-tailed test

        Returns:
            Required sample size per group
        """
        z_alpha = norm.ppf(1 - alpha / 2) if two_sided else norm.ppf(1 - alpha)
        z_beta = norm.ppf(power)
        p_pool = (p1 + p2) / 2
        n = ((z_alpha * np.sqrt(2 * p_pool * (1 - p_pool)) +
              z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p1 - p2) ** 2
        return int(np.ceil(n))

    @staticmethod
    def mde(n: int, p1: float, alpha: float = 0.05, power: float = 0.80) -> float:
        """Minimum detectable effect (absolute difference in proportions)."""
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        p_pool = p1  # approximate
        mde = (z_alpha + z_beta) * np.sqrt(2 * p_pool * (1 - p_pool) / n)
        return float(mde)


class ABTester:
    """Hypothesis tests for A/B experiments."""

    @staticmethod
    def two_proportion_z_test(x1: int, n1: int, x2: int, n2: int) -> dict:
        """
        Two-proportion z-test (pooled).

        Args:
            x1, n1: successes and total in group 1
            x2, n2: successes and total in group 2

        Returns:
            dict with z-statistic, p-value, effect size (Cohen's h)
        """
        p1, p2 = x1 / n1, x2 / n2
        p_pool = (x1 + x2) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        z = (p1 - p2) / se
        p_value = 2 * (1 - norm.cdf(abs(z)))

        # Cohen's h
        h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

        return {"z": z, "p_value": p_value, "cohens_h": h,
                "p1": p1, "p2": p2}

    @staticmethod
    def welch_t_test(group1: np.ndarray, group2: np.ndarray) -> dict:
        """Welch's t-test (unequal variances)."""
        t, p = stats.ttest_ind(group1, group2, equal_var=False)
        # Cohen's d
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * group1.std(ddof=1)**2 +
                               (n2 - 1) * group2.std(ddof=1)**2) / (n1 + n2 - 2))
        d = (group1.mean() - group2.mean()) / pooled_std
        return {"t": t, "p_value": p, "cohens_d": d}

    @staticmethod
    def log_rank_test(time1, event1, time2, event2) -> dict:
        """Log-rank test for survival data."""
        kmf1 = KaplanMeierFitter().fit(time1, event1)
        kmf2 = KaplanMeierFitter().fit(time2, event2)
        # lifelines handles the log-rank test
        from lifelines.statistics import logrank_test
        result = logrank_test(time1, time2, event_observed_A=event1, event_observed_B=event2)
        return {"test_statistic": result.test_statistic,
                "p_value": result.p_value,
                "median_survival_1": kmf1.median_survival_time_,
                "median_survival_2": kmf2.median_survival_time_}

    @staticmethod
    def cox_ph(df: pd.DataFrame, duration_col: str, event_col: str,
               covariates: list) -> dict:
        """
        Cox Proportional Hazards model.

        Args:
            df: DataFrame with duration, event, and covariate columns
            duration_col: time-to-event column
            event_col: 1 if event occurred, 0 if censored
            covariates: list of column names to use as predictors

        Returns:
            dict with hazard ratios, p-values, and model summary
        """
        cph = CoxPHFitter()
        cph.fit(df, duration_col=duration_col, event_col=event_col)
        summary = cph.summary

        results = {}
        for cov in covariates:
            if cov in summary.index:
                results[cov] = {
                    "hazard_ratio": summary.loc[cov, "exp(coef)"],
                    "p_value": summary.loc[cov, "p"],
                    "ci_lower": summary.loc[cov, "exp(coef) lower"],
                    "ci_upper": summary.loc[cov, "exp(coef) upper"],
                }
        return results


class PropensityScoreMatcher:
    """
    Propensity score matching for observational studies.
    Used when you can't randomize (e.g., "customers who got AI agent vs. not").
    """

    def __init__(self, caliper: float = 0.2):
        """
        Args:
            caliper: maximum distance (in logit scale) for a valid match.
                     0.2 is standard (Rosenbaum & Rubin, 1983).
        """
        self.caliper = caliper
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X: pd.DataFrame, treatment: np.ndarray) -> "PropensityScoreMatcher":
        """Fit logistic regression to estimate propensity scores."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, treatment)
        self.propensity_scores = self.model.predict_proba(X_scaled)[:, 1]
        return self

    def match(self, treatment: np.ndarray) -> tuple:
        """
        Nearest-neighbor matching with caliper.

        Returns:
            (treated_indices, control_indices) — matched pairs
        """
        treated = np.where(treatment == 1)[0]
        control = np.where(treatment == 0)[0]

        matched_treated = []
        matched_control = []
        used_control = set()

        for t_idx in treated:
            t_score = self.propensity_scores[t_idx]
            # Find nearest unmatched control within caliper
            distances = np.abs(self.propensity_scores[control] - t_score)
            for c_pos, c_idx in enumerate(control):
                if c_idx in used_control:
                    continue
                if distances[c_pos] <= self.caliper:
                    matched_treated.append(t_idx)
                    matched_control.append(c_idx)
                    used_control.add(c_idx)
                    break

        return np.array(matched_treated), np.array(matched_control)


class SequentialMonitor:
    """
    Monitor A/B test at interim looks with Bonferroni alpha spending.
    Prevents Type I error inflation from peeking at results.
    """

    def __init__(self, n_looks: int, alpha: float = 0.05):
        """
        Args:
            n_looks: total number of planned interim looks (including final)
            alpha: total Type I error budget
        """
        self.n_looks = n_looks
        self.alpha_per_look = alpha / n_looks  # Bonferroni
        self.looks_done = 0

    def check(self, test_statistic: float, two_sided: bool = True) -> dict:
        """
        Check if the current look is significant.

        Args:
            test_statistic: z-statistic from the current test

        Returns:
            dict with decision and critical value
        """
        self.looks_done += 1
        if two_sided:
            critical = norm.ppf(1 - self.alpha_per_look / 2)
        else:
            critical = norm.ppf(1 - self.alpha_per_look)

        significant = abs(test_statistic) > critical
        return {
            "look": self.looks_done,
            "critical_value": critical,
            "test_statistic": test_statistic,
            "significant": significant,
            "stop": significant or self.looks_done == self.n_looks
        }   
