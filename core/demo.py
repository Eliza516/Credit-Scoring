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
