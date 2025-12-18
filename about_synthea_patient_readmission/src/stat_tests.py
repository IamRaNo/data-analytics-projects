from scipy.stats import (
    levene,
    ttest_ind,
    chi2_contingency,
    shapiro,
    mannwhitneyu,
    fisher_exact
)
import numpy as np


# ==========================================================
# INTERNAL HELPERS
# ==========================================================

def _confidence_label(n1, n2):
    total = n1 + n2
    smaller = min(n1, n2)

    if smaller >= 0.10 * total:
        return "High confidence"
    elif smaller >= 0.05 * total:
        return "Moderate confidence"
    else:
        return "Low confidence"


def _check_normality(g1, g2):
    if len(g1) > 5000 or len(g2) > 5000:
        return True, "Large sample (CLT assumed)"

    p1 = shapiro(g1)[1]
    p2 = shapiro(g2)[1]

    return (p1 >= 0.05 and p2 >= 0.05), "Shapiro-Wilk test"


def _check_variance(g1, g2):
    p = levene(g1, g2)[1]
    return p >= 0.05


def _cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)

    pooled_std = np.sqrt(
        ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
    )

    d = (np.mean(g1) - np.mean(g2)) / pooled_std
    ad = abs(d)

    if ad < 0.2:
        return d, "Negligible"
    elif ad < 0.5:
        return d, "Small"
    elif ad < 0.8:
        return d, "Medium"
    else:
        return d, "Large"


def _rank_biserial(u, n1, n2):
    r = 1 - (2 * u) / (n1 * n2)
    ar = abs(r)

    if ar < 0.1:
        return r, "Very Weak"
    elif ar < 0.3:
        return r, "Weak"
    elif ar < 0.5:
        return r, "Medium"
    else:
        return r, "Strong"


# ==========================================================
# NUMERICAL COMPARISON
# ==========================================================

def compare_means(data, group_col, value_col, focus_value):
    g1 = data[data[group_col] == focus_value][value_col].dropna()
    g2 = data[data[group_col] != focus_value][value_col].dropna()

    if len(g1) < 2 or len(g2) < 2:
        print("❌ Not enough data for comparison.")
        return

    print(f"\n=== Comparing '{value_col}' by '{group_col}' ===")
    print(f"Group '{focus_value}': n={len(g1)} | Others: n={len(g2)}")

    confidence = _confidence_label(len(g1), len(g2))
    is_normal, norm_reason = _check_normality(g1, g2)

    if is_normal:
        equal_var = _check_variance(g1, g2)
        stat, p = ttest_ind(g1, g2, equal_var=equal_var)
        eff, strength = _cohens_d(g1, g2)

        test_name = "T-Test"
        eff_name = "Cohen's d"
        reason = f"normal data ({norm_reason})"

    else:
        stat, p = mannwhitneyu(g1, g2, alternative="two-sided")
        eff, strength = _rank_biserial(stat, len(g1), len(g2))

        test_name = "Mann-Whitney U"
        eff_name = "Rank-Biserial"
        reason = "non-normal data"

    print(f"Test Used:      {test_name} ({reason})")
    print(f"P-Value:        {p:.5f}")
    print(f"Confidence:     {confidence}")

    if p < 0.05:
        print("Verdict:        Significant difference ✅")
        print(f"Effect Size:    {eff_name} = {eff:.3f} ({strength})")

        impact = {
            "Negligible": "No practical impact",
            "Small": "Minor impact",
            "Medium": "Meaningful impact",
            "Large": "Strong practical impact",
            "Very Weak": "Minimal impact",
            "Weak": "Limited impact",
            "Strong": "Strong impact"
        }.get(strength, "Moderate impact")

        print(f"Practical Impact: {impact}")
    else:
        print("Verdict:        No significant difference ❌")

    print("=" * 40)


# ==========================================================
# CATEGORICAL ASSOCIATION
# ==========================================================

def test_association(ct):
    print("\n=== Categorical Association Test ===")

    stat, p, dof, expected = chi2_contingency(ct)
    min_expected = expected.min()

    test_name = "Chi-Square Test"
    if min_expected < 5:
        if ct.shape == (2, 2):
            _, p = fisher_exact(ct)
            test_name = "Fisher's Exact Test"
        else:
            print(f"⚠️ Low expected counts (min={min_expected:.2f})")

    n = ct.to_numpy().sum()
    r, k = ct.shape
    cramer_v = np.sqrt(stat / (n * (min(r, k) - 1))) if n > 0 else 0

    if cramer_v < 0.1:
        strength = "Weak"
    elif cramer_v < 0.3:
        strength = "Moderate"
    else:
        strength = "Strong"

    confidence = _confidence_label(
        ct.sum(axis=1).min(),
        ct.sum(axis=1).max()
    )

    print(f"Test Used:      {test_name}")
    print(f"P-Value:        {p:.5f}")
    print(f"Confidence:     {confidence}")

    if p < 0.05:
        print("Verdict:        Significant association ✅")
        print(f"Effect Size:    Cramér's V = {cramer_v:.3f} ({strength})")

        impact = {
            "Weak": "Limited influence",
            "Moderate": "Noticeable influence",
            "Strong": "Strong influencing factor"
        }[strength]

        print(f"Practical Impact: {impact}")
        print(f"Conclusion:     Association exists with {strength.lower()} strength.")
    else:
        print("Verdict:        No association ❌")
        print("Conclusion:     No evidence of relationship.")

    print("=" * 40)
