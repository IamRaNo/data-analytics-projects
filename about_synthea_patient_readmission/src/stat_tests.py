from scipy.stats import levene, ttest_ind, chi2_contingency, shapiro, mannwhitneyu, fisher_exact
import numpy as np
import pandas as pd

# ==========================================
# 1. HELPER FUNCTIONS (INTERNAL LOGIC)
# ==========================================

def _check_normality(g1, g2):
    n1, n2 = len(g1), len(g2)
    
    # Large sample optimization
    if n1 > 5000 or n2 > 5000:
        return True, "Sample size > 5000 (CLT assumed)"
    
    # Shapiro-Wilk test
    stat1, p1 = shapiro(g1)
    stat2, p2 = shapiro(g2)
    
    is_normal = (p1 >= 0.05) and (p2 >= 0.05)
    return is_normal, "Shapiro-Wilk Test"

def _check_variance(g1, g2):
    stat, p = levene(g1, g2)
    equal_var = p >= 0.05
    return equal_var, p

def _calculate_cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    
    # Pooled Standard Deviation
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    d = (np.mean(g1) - np.mean(g2)) / pooled_std
    
    abs_d = abs(d)
    if abs_d < 0.2: strength = "Negligible"
    elif abs_d < 0.5: strength = "Small"
    elif abs_d < 0.8: strength = "Medium"
    else: strength = "Large"
    
    return d, strength

def _calculate_rank_biserial(u_stat, n1, n2):
    rbc = 1 - (2 * u_stat) / (n1 * n2)
    abs_rbc = abs(rbc)
    
    if abs_rbc < 0.1: strength = "Very Weak"
    elif abs_rbc < 0.3: strength = "Weak"
    elif abs_rbc < 0.5: strength = "Medium"
    else: strength = "Strong"
    
    return rbc, strength

# ==========================================
# 2. MAIN RUNNER FUNCTIONS
# ==========================================

def compare_means(data, group_col, value_col, val):
    # 1. Prepare Data
    g1 = data[data[group_col] == val][value_col].dropna()
    g2 = data[data[group_col] != val][value_col].dropna()
    
    if len(g1) < 2 or len(g2) < 2:
        print("❌ Error: Not enough data points to perform comparison.")
        return

    print(f"\n=== Comparing '{value_col}' by Group '{group_col}' ===")
    print(f"Group 1 ({val}): n={len(g1)} | Group 2 (Others): n={len(g2)}")

    # 2. Check Assumptions
    is_normal, norm_reason = _check_normality(g1, g2)
    
    # 3. Choose and Run Test
    if is_normal:
        # --- PARAMETRIC PATH (T-TEST) ---
        equal_var, lev_p = _check_variance(g1, g2)
        var_status = "Equal" if equal_var else "Unequal"
        
        stat, p = ttest_ind(g1, g2, equal_var=equal_var)
        effect_val, effect_str = _calculate_cohens_d(g1, g2)
        
        test_name = f"T-Test ({var_status} Variance)"
        effect_name = "Cohen's d"
        
    else:
        # --- NON-PARAMETRIC PATH (MANN-WHITNEY) ---
        stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
        effect_val, effect_str = _calculate_rank_biserial(stat, len(g1), len(g2))
        
        test_name = "Mann-Whitney U Test"
        effect_name = "Rank-Biserial"

    # 4. Print Report
    print(f"Test Used:      {test_name}")
    print(f"Normality:      {norm_reason} -> {'Pass' if is_normal else 'Fail'}")
    print(f"P-Value:        {p:.5f}")
    
    if p < 0.05:
        print(f"Verdict:        Significant Difference ✅")
        print(f"Effect Size:    {effect_name} = {effect_val:.3f} ({effect_str})")
    else:
        print(f"Verdict:        No Significant Difference ❌")

    print("="*20)

def test_association(ct):
    print("\n=== Categorical Association Test ===")
    
    # 1. Run Basic Chi-Square
    stat, p, dof, expected = chi2_contingency(ct)
    
    # 2. Check Assumptions (Rule of 5)
    min_expected = expected.min()
    sample_issue = min_expected < 5
    
    test_used = "Chi-Square Test"
    
    # 3. Fallback logic for small samples
    if sample_issue:
        if ct.shape == (2, 2):
            _, p = fisher_exact(ct)
            test_used = "Fisher's Exact Test (Small Sample Correction)"
        else:
            print(f"⚠️ Warning: Low expected frequencies (min={min_expected:.2f}). Result may be unstable.")

    # 4. Calculate Strength (Cramer's V)
    n = ct.sum().sum()
    r, k = ct.shape
    if n > 0 and min(r, k) > 1:
        cramer_v = np.sqrt(stat / (n * (min(r, k) - 1)))
    else:
        cramer_v = 0.0
        
    if cramer_v < 0.1: strength = "Weak"
    elif cramer_v < 0.3: strength = "Moderate"
    else: strength = "Strong"
    
    # 5. Print Report
    print(f"Test Used:      {test_used}")
    print(f"P-Value:        {p:.5f}")
    
    if p < 0.05:
        print(f"Verdict:        Significant Dependence ✅")
        print(f"Strength:       Cramer's V = {cramer_v:.3f} ({strength})")
    else:
        print(f"Verdict:        Independent (No Relationship) ❌")
    
    print("="*20)
