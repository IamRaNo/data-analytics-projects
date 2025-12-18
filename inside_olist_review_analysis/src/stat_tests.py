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

def _check_balance(n1, n2):
    total = n1 + n2
    smaller = min(n1, n2)
    ratio = smaller / total if total > 0 else 0

    if smaller > 1000:
        return "Robust (High N)"
    
    if ratio >= 0.10:
        return "Balanced"
    elif ratio >= 0.05:
        return "Moderate Imbalance"
    else:
        return "High Imbalance (Caution)"

def _check_normality(g1, g2):
    if len(g1) > 5000 or len(g2) > 5000: return True
    p1, p2 = shapiro(g1)[1], shapiro(g2)[1]
    return (p1 >= 0.05 and p2 >= 0.05)

def _check_variance(g1, g2):
    return levene(g1, g2)[1] >= 0.05

def _get_strength_numerical(g1, g2, is_normal):
    # Calculate Effect Size
    if is_normal:
        # Cohen's d
        n1, n2 = len(g1), len(g2)
        s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        eff = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std != 0 else 0
        metric = "Cohen's d"
        cutoff = 0.5
    else:
        # Rank-Biserial
        u, _ = mannwhitneyu(g1, g2, alternative="two-sided")
        eff = 1 - (2 * u) / (len(g1) * len(g2))
        metric = "Rank-Biserial"
        cutoff = 0.3

    # Determine Impact Label
    ae = abs(eff)
    if ae < 0.1: impact = "Negligible"
    elif ae < cutoff: impact = "Small/Weak"
    elif ae < (cutoff + 0.3): impact = "Moderate"
    else: impact = "Strong"
    
    return eff, metric, impact

# ==========================================================
# 1. NUMERICAL COMPARISON (Clean)
# ==========================================================

def compare_means(data, group_col, value_col, focus_value):
    g1 = data[data[group_col] == focus_value][value_col].dropna()
    g2 = data[data[group_col] != focus_value][value_col].dropna()

    if len(g1) < 2 or len(g2) < 2: return

    print(f"\n--- Comparing '{value_col}' by '{group_col}' ({focus_value} vs Others) ---")
    
    # 1. Run Test
    is_normal = _check_normality(g1, g2)
    if is_normal:
        equal_var = _check_variance(g1, g2)
        _, p = ttest_ind(g1, g2, equal_var=equal_var)
    else:
        _, p = mannwhitneyu(g1, g2, alternative="two-sided")

    # 2. Get Strength
    eff, metric, impact = _get_strength_numerical(g1, g2, is_normal)
    
    # 3. Check Balance
    balance = _check_balance(len(g1), len(g2))

    # --- THE 3 OUTPUTS ---
    # Output 1: Verdict
    if p < 0.05:
        print(f"1. Verdict:    ✅ SIGNIFICANT difference (p={p:.5f})")
    else:
        print(f"1. Verdict:    ❌ NOT Significant (p={p:.5f})")

    # Output 2: Strength
    print(f"2. Strength:   {impact} ({metric} = {eff:.3f})")

    # Output 3: Reliability
    print(f"3. Balance:    {balance} (n={len(g1)} vs n={len(g2)})")
    print("-" * 60)


# ==========================================================
# 2. CATEGORICAL ASSOCIATION (Clean)
# ==========================================================

def test_association(ct):
    print(f"\n--- Association Test: {ct.index.name} vs {ct.columns.name} ---")

    # 1. Run Test
    stat, p, _, expected = chi2_contingency(ct)
    if expected.min() < 5 and ct.shape == (2, 2):
        _, p = fisher_exact(ct)

    # 2. Get Strength (Cramer's V)
    n = ct.to_numpy().sum()
    r, k = ct.shape
    cramer = np.sqrt(stat / (n * (min(r, k) - 1))) if n > 0 else 0
    
    if cramer < 0.1: impact = "Weak"
    elif cramer < 0.3: impact = "Moderate"
    else: impact = "Strong"

    # 3. Check Balance
    row_sums = ct.sum(axis=1)
    balance = _check_balance(row_sums.min(), row_sums.max())

    # --- THE 3 OUTPUTS ---
    # Output 1: Verdict
    if p < 0.05:
        print(f"1. Verdict:    ✅ SIGNIFICANT Association (p={p:.5f})")
    else:
        print(f"1. Verdict:    ❌ No Association (p={p:.5f})")

    # Output 2: Strength
    print(f"2. Strength:   {impact} (Cramer's V = {cramer:.3f})")

    # Output 3: Reliability
    print(f"3. Balance:    {balance}")
    print("-" * 60)