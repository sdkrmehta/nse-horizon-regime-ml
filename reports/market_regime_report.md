# Regime Detection Report

**Generated**: 2026-09-15 15:29:30
**Method**: 20-day rolling realized volatility with quantile thresholds (33rd/67th percentile)
**Regime Definition**: Low (≤ Q33), Medium (Q33–Q67), High (> Q67)
**Leakage Control**: Thresholds computed from training data ONLY per split

---

## Per-Split Regime Summary

### split_1

- **Training period**: 2015-01-01 → 2019-12-31
- **Test period**: 2020-01-01 → 2020-12-31
- **Thresholds**: Low ≤ 0.1993, High > 0.2676

| Regime | Test Count | Test % |
|--------|-----------|--------|
| Low | 477 | 12.7% |
| Medium | 782 | 20.8% |
| High | 2506 | 66.6% |
| **Total** | **3765** | **100%** |

---

### split_2

- **Training period**: 2015-01-01 → 2020-12-31
- **Test period**: 2021-01-01 → 2021-12-31
- **Thresholds**: Low ≤ 0.2052, High > 0.2839

| Regime | Test Count | Test % |
|--------|-----------|--------|
| Low | 1163 | 31.3% |
| Medium | 1317 | 35.4% |
| High | 1240 | 33.3% |
| **Total** | **3720** | **100%** |

---

### split_3

- **Training period**: 2015-01-01 → 2021-12-31
- **Test period**: 2022-01-01 → 2022-12-31
- **Thresholds**: Low ≤ 0.2056, High > 0.2840

| Regime | Test Count | Test % |
|--------|-----------|--------|
| Low | 1048 | 28.2% |
| Medium | 1606 | 43.2% |
| High | 1066 | 28.7% |
| **Total** | **3720** | **100%** |

---

### split_4

- **Training period**: 2015-01-01 → 2022-12-31
- **Test period**: 2023-01-01 → 2023-12-31
- **Thresholds**: Low ≤ 0.2067, High > 0.2828

| Regime | Test Count | Test % |
|--------|-----------|--------|
| Low | 2850 | 77.6% |
| Medium | 690 | 18.8% |
| High | 135 | 3.7% |
| **Total** | **3675** | **100%** |

---

### split_5

- **Training period**: 2015-01-01 → 2023-12-31
- **Test period**: 2024-01-01 → 2024-12-31
- **Thresholds**: Low ≤ 0.1982, High > 0.2726

| Regime | Test Count | Test % |
|--------|-----------|--------|
| Low | 1406 | 38.1% |
| Medium | 1521 | 41.2% |
| High | 763 | 20.7% |
| **Total** | **3690** | **100%** |

---

### split_6_final

- **Training period**: 2015-01-01 → 2024-12-31
- **Test period**: 2025-01-01 → 2025-06-30
- **Thresholds**: Low ≤ 0.1973, High > 0.2689

| Regime | Test Count | Test % |
|--------|-----------|--------|
| Low | 719 | 39.3% |
| Medium | 655 | 35.8% |
| High | 456 | 24.9% |
| **Total** | **1830** | **100%** |

---

## Threshold Stability Across Splits

| Split | Low Threshold | High Threshold |
|-------|--------------|----------------|
| split_1 | 0.1993 | 0.2676 |
| split_2 | 0.2052 | 0.2839 |
| split_3 | 0.2056 | 0.2840 |
| split_4 | 0.2067 | 0.2828 |
| split_5 | 0.1982 | 0.2726 |
| split_6_final | 0.1973 | 0.2689 |
