"""Dữ liệu demo tổng hợp (không phải dữ liệu thật) để người dùng thử."""
from __future__ import annotations

import numpy as np
import pandas as pd

DEMO_MAPPING = {
    "id": "loan_id",
    "open_date": "open_date",
    "dpd_max": "max_dpd",
    "first_bad_mob": "first_bad_mob",
    "months_observed": "months_observed",
    "n_payments": "n_payments",
    "charge_off_flag": "charge_off",
    "fraud_flag": "fraud_flag",
    "staff_vip_flag": "staff_flag",
}


def make_demo(n: int = 40000, seed: int = 42, obs_end: str = "2025-12-31", target_rate: float = 0.17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-06-30")
    open_date = start + pd.to_timedelta(rng.integers(0, (end - start).days, n), unit="D")
    obs = pd.Timestamp(obs_end)
    mob_obs = ((obs.year - open_date.year) * 12 + (obs.month - open_date.month)).to_numpy()
    mob_obs = np.minimum(mob_obs, 36)

    product = rng.choice(["personal", "motorbike", "credit_card"], n, p=[0.5, 0.3, 0.2])
    channel = rng.choice(["branch", "online", "agent"], n, p=[0.4, 0.35, 0.25])
    gender = rng.choice(["M", "F"], n)
    marital = rng.choice(["single", "married", "other"], n, p=[0.4, 0.5, 0.1])
    zipc = rng.choice(["70000", "10000", "50000", "90000"], n, p=[0.35, 0.35, 0.2, 0.1])
    device = rng.choice(["ios", "android", "other"], n, p=[0.3, 0.6, 0.1])

    age = np.clip(rng.normal(36, 9, n), 20, 68).round()
    income = np.exp(rng.normal(16.4, 0.5, n)) / 1e6          # triệu VND
    emp_years = np.clip(age - 21 - rng.exponential(5, n), 0, 40).round(1)
    loan_amt = income * rng.uniform(3, 12, n)
    term = rng.choice([12, 24, 36, 48], n)
    dti = np.clip(rng.beta(2, 5, n) * 1.2, 0, 1.5)
    util = np.clip(rng.beta(2, 3, n), 0, 1)
    inq = rng.poisson(1.5, n)
    delinq = rng.poisson(0.3, n)
    bureau_m = np.where(rng.random(n) < 0.18, np.nan, np.clip(rng.exponential(36, n), 1, 240).round())
    thin = np.isnan(bureau_m)

    z = 1.7 * (1.6 * dti + 1.2 * util + 0.22 * inq + 0.5 * delinq
         - 0.35 * np.log(income + 1) + 0.012 * (36 - age) - 0.03 * emp_years
         + 0.25 * thin + 0.25 * (product == "credit_card") + 0.2 * (zipc == "90000") - 0.12 * (zipc == "70000")
         + 0.2 * (channel == "agent")) + rng.normal(0, 0.35, n)
    lo, hi = -30.0, 30.0                                   # chọn intercept để tỷ lệ default ≈ target_rate
    for _ in range(50):
        mid_c = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid_c)))).mean() > target_rate:
            hi = mid_c
        else:
            lo = mid_c
    p = 1 / (1 + np.exp(-(z + (lo + hi) / 2)))
    default = rng.random(n) < p
    first_bad = np.where(default, np.clip(rng.gamma(3.0, 1.8, n).round() + 2, 3, 30), np.nan)
    first_bad = np.where(first_bad <= mob_obs, first_bad, np.nan)   # chỉ thấy bad đã xảy ra
    is_bad = ~np.isnan(first_bad)
    mid = (~is_bad) & (rng.random(n) < 0.10 + 0.25 * p)             # từng quá hạn nhẹ
    max_dpd = np.where(is_bad, rng.choice([91, 120, 150, 180], n, p=[0.5, 0.25, 0.15, 0.1]),
                       np.where(mid, rng.choice([15, 30, 45, 60, 80], n), 0))
    charge_off = (is_bad & (rng.random(n) < 0.35)).astype(int)
    n_pay = np.minimum(mob_obs, term) - np.where(is_bad, 0, 0)
    n_pay = np.maximum(n_pay, 0)

    df = pd.DataFrame({
        "loan_id": np.arange(1, n + 1),
        "open_date": open_date.normalize(),
        "product_type": product, "channel": channel,
        "age": age, "income_mil": income.round(2), "employment_years": emp_years,
        "loan_amount_mil": loan_amt.round(1), "term_months": term,
        "dti": dti.round(3), "credit_util": util.round(3),
        "inquiries_6m": inq, "delinq_12m": delinq, "bureau_months": bureau_m,
        "gender": gender, "marital_status": marital, "zip_code": zipc, "device_type": device,
        "months_observed": mob_obs, "n_payments": n_pay,
        "first_bad_mob": first_bad, "max_dpd": max_dpd, "charge_off": charge_off,
        "fraud_flag": (rng.random(n) < 0.004).astype(int),
        "staff_flag": (rng.random(n) < 0.01).astype(int),
    })
    # giá trị đặc biệt & missing & outlier để demo làm sạch
    df.loc[rng.random(n) < 0.03, "income_mil"] = -999
    df.loc[rng.random(n) < 0.05, "employment_years"] = np.nan
    df.loc[rng.random(n) < 0.002, "age"] = 150
    df["all_const"] = 1
    df["mostly_missing"] = np.where(rng.random(n) < 0.75, np.nan, rng.normal(size=n))
    return df


def make_demo_performance(df: pd.DataFrame, max_mob: int = 24, seed: int = 7) -> pd.DataFrame:
    """Panel performance dọc (loan_id, snapshot_month, dpd, balance) nhất quán với df demo:
    khoản bad chạm DPD ≥ 91 đúng tại first_bad_mob; khoản 'quá hạn nhẹ' có 1 tháng DPD 15–80; còn lại DPD 0."""
    rng = np.random.default_rng(seed)
    obs = np.minimum(df["months_observed"].to_numpy().astype(int), max_mob)
    obs = np.maximum(obs, 0)
    idx = np.repeat(np.arange(len(df)), obs)
    start = np.cumsum(obs) - obs
    m = np.arange(obs.sum()) - np.repeat(start, obs) + 1            # MOB 1..obs của từng khoản
    fb = df["first_bad_mob"].to_numpy(dtype=float)[idx]
    mx = df["max_dpd"].to_numpy(dtype=float)[idx]
    dpd = np.zeros(len(m))
    bad = ~np.isnan(fb) & (m >= fb)
    dpd[bad] = np.minimum(mx[bad], 91 + 30 * (m[bad] - fb[bad]))
    ramp = ~np.isnan(fb) & (m < fb)
    dpd[ramp & (m == fb - 1)] = 60
    dpd[ramp & (m == fb - 2)] = 30
    mid = np.isnan(df["first_bad_mob"].to_numpy(dtype=float)) & (df["max_dpd"].to_numpy() > 0)
    pick = (rng.random(len(df)) * np.maximum(obs, 1)).astype(int) + 1   # tháng có DPD nhẹ
    hit = mid[idx] & (m == pick[idx])
    dpd[hit] = mx[hit]
    term = df["term_months"].to_numpy(dtype=float)[idx]
    amt = df["loan_amount_mil"].to_numpy(dtype=float)[idx] * 1e6
    bal = np.maximum(amt * (1 - m / np.maximum(term, 1)), 0.05 * amt).round(0)
    opens = pd.to_datetime(df["open_date"]).to_numpy()[idx].astype("datetime64[M]")
    snap = (opens + m).astype("datetime64[M]")
    return pd.DataFrame({"loan_id": df["loan_id"].to_numpy()[idx], "snapshot_month": pd.to_datetime(snap) + pd.offsets.MonthEnd(0),
                         "dpd": dpd.astype(int), "balance": bal})


def make_lending_club_like(n: int = 20000, seed: int = 11) -> pd.DataFrame:
    """Dữ liệu MÔ PHỎNG có cấu trúc giống Lending Club (không phải dữ liệu thật): cột trạng thái chữ, ngày dạng 'Dec-2015',
    kỳ hạn '36 months', cột chỉ có sau khi vay (rò rỉ), cột nhiều giá trị chữ… để thử hồ sơ 'lending_club' khi chưa có file thật."""
    rng = np.random.default_rng(seed)
    issue = pd.Timestamp("2008-01-01") + pd.to_timedelta(rng.integers(0, 365 * 9, n), unit="D")
    issue = issue.to_period("M").to_timestamp()
    term = rng.choice([36, 60], n, p=[0.75, 0.25])
    grade = rng.choice(list("ABCDEFG"), n, p=[0.17, 0.30, 0.28, 0.14, 0.07, 0.03, 0.01])
    gi = np.array([ord(g) - 65 for g in grade])
    int_rate = np.round(6 + 2.4 * gi + rng.normal(0, 1.2, n), 2)
    fico = np.round(np.clip(740 - 12 * gi + rng.normal(0, 25, n), 620, 840) / 5) * 5
    dti = np.clip(rng.normal(17, 8, n), 0, 60).round(2)
    income = np.exp(rng.normal(11.1, 0.55, n)).round(-2)
    loan = (np.clip(rng.normal(14000, 8000, n), 1000, 40000) // 25 * 25).astype(float)
    home = rng.choice(["RENT", "MORTGAGE", "OWN"], n, p=[0.45, 0.42, 0.13])
    purpose = rng.choice(["debt_consolidation", "credit_card", "home_improvement", "small_business", "other", "car"], n, p=[0.55, 0.2, 0.08, 0.04, 0.09, 0.04])
    emp_len = rng.choice(["< 1 year", "1 year", "2 years", "3 years", "5 years", "10+ years", None], n, p=[0.08, 0.07, 0.1, 0.1, 0.1, 0.5, 0.05])
    earliest = issue - pd.to_timedelta((rng.gamma(4, 1200, n) + 800).astype(int), unit="D")
    revol_util = np.clip(rng.normal(52, 25, n), 0, 120).round(1)
    z = (-4.0 + 0.35 * gi + 0.025 * (dti - 17) - 0.008 * (fico - 700) + 0.5 * (term == 60) - 0.35 * np.log(income / 60000)
         + 0.004 * (revol_util - 50) + 0.25 * (purpose == "small_business") + rng.normal(0, 0.4, n))
    lo, hi = -20.0, 20.0                                            # chọn hệ số chặn để tỷ lệ Charged Off ≈ 18%
    for _ in range(50):
        mid_c = (lo + hi) / 2
        lo, hi = (lo, mid_c) if (1 / (1 + np.exp(-(z + mid_c)))).mean() > 0.18 else (mid_c, hi)
    p_bad = 1 / (1 + np.exp(-(z + (lo + hi) / 2)))
    bad = rng.random(n) < p_bad
    end = pd.Timestamp("2018-12-01")
    due = issue + pd.to_timedelta(term * 30.4, unit="D")
    matured = due <= end
    paid = ~bad
    status = np.where(bad, "Charged Off", "Fully Paid").astype(object)
    cur = (~matured) & (rng.random(n) < 0.8)                       # khoản chưa đáo hạn đa số còn "Current"
    status[cur] = "Current"
    late = (~cur) & (rng.random(n) < 0.01)
    status[late] = rng.choice(["Late (31-120 days)", "In Grace Period", "Default"], late.sum())
    life = np.where(bad, rng.integers(5, 30, n), term - rng.integers(0, 12, n))
    lp = pd.Series(issue + pd.to_timedelta(life * 30.4, unit="D"))
    last_pay = lp.where(lp <= end, end).dt.to_period("M").dt.to_timestamp()
    last_pay = last_pay.where(~cur, end)
    fmt = lambda s: pd.Series(s).dt.strftime("%b-%Y").to_numpy()  # noqa: E731
    return pd.DataFrame({
        "id": np.arange(40_000_000, 40_000_000 + n).astype(str), "member_id": np.nan, "loan_amnt": loan, "funded_amnt": loan,
        "term": np.where(term == 36, " 36 months", " 60 months"), "int_rate": int_rate, "grade": grade,
        "emp_title": rng.choice([f"title_{i}" for i in range(400)], n), "emp_length": emp_len, "home_ownership": home, "annual_inc": income,
        "issue_d": fmt(issue), "loan_status": status, "purpose": purpose, "zip_code": rng.choice([f"{i:03d}xx" for i in range(100, 900)], n),
        "addr_state": rng.choice(["CA", "NY", "TX", "FL", "IL", "NJ", "PA", "OH"], n), "dti": dti, "earliest_cr_line": fmt(earliest),
        "fico_range_low": fico, "fico_range_high": fico + 4, "revol_util": revol_util, "open_acc": rng.poisson(11, n).astype(float),
        "pub_rec": rng.poisson(0.2, n).astype(float), "inq_last_6mths": rng.poisson(0.9, n).astype(float),
        "total_pymnt": np.where(bad, loan * rng.uniform(0.2, 0.7, n), loan * rng.uniform(1.05, 1.25, n)).round(2),
        "recoveries": np.where(bad, loan * rng.uniform(0, 0.1, n), 0.0).round(2), "out_prncp": np.where(cur, loan * 0.4, 0.0),
        "last_pymnt_d": fmt(last_pay), "last_fico_range_high": np.where(bad, fico - 40, fico + 5),
        "all_empty_2012": np.nan,
    })
