from src.config import (
    RANDOM_SEED, N_SAMPLES, TARGET_POSITIVE_RATE, TREATMENT_ARMS,
    DATA_FILE, COST_ASSUMPTIONS, CLINICAL_RANGES
)
import os
import numpy as np
import pandas as pd
from scipy import stats

import sys
from src.logger import get_logger
logger = get_logger(__name__)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np.random.seed(RANDOM_SEED)

def generate_demographics(n: int) -> pd.DataFrame:
    age = np.clip(
        np.random.normal(loc=55, scale=16, size=n),
        CLINICAL_RANGES["age"][0], CLINICAL_RANGES["age"][1]
    ).astype(int)

    sex = np.random.choice(["Male", "Female"], size=n, p=[0.52, 0.48])

    height = np.where(
        sex == "Male",
        np.random.normal(175, 8, n),
        np.random.normal(162, 7, n)
    )
    height = np.clip(
        height, CLINICAL_RANGES["height_cm"][0], CLINICAL_RANGES["height_cm"][1])

    bmi_base = np.random.lognormal(mean=np.log(26), sigma=0.2, size=n)
    bmi_base += (age - 40) * 0.03
    weight = bmi_base * (height / 100) ** 2
    weight = np.clip(
        weight, CLINICAL_RANGES["weight_kg"][0], CLINICAL_RANGES["weight_kg"][1])

    return pd.DataFrame({
        "age": age,
        "sex": sex,
        "height_cm": np.round(height, 1),
        "weight_kg": np.round(weight, 1),
    })

def generate_clinical_markers(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    bmi = df["weight_kg"] / (df["height_cm"] / 100) ** 2
    age = df["age"].values

    sbp = 110 + 0.5 * age + 0.8 * bmi + np.random.normal(0, 12, n)
    sbp = np.clip(sbp, CLINICAL_RANGES["sbp"][0], CLINICAL_RANGES["sbp"][1])

    dbp = 30 + 0.4 * sbp + np.random.normal(0, 8, n)
    dbp = np.clip(dbp, CLINICAL_RANGES["dbp"][0], CLINICAL_RANGES["dbp"][1])

    hba1c = 4.5 + 0.015 * age + 0.04 * bmi + np.random.exponential(0.3, n)
    hba1c = np.clip(
        hba1c, CLINICAL_RANGES["hba1c"][0], CLINICAL_RANGES["hba1c"][1])

    ldl = 80 + 0.3 * age + 0.6 * bmi + np.random.normal(0, 25, n)
    ldl = np.clip(ldl, CLINICAL_RANGES["ldl"][0], CLINICAL_RANGES["ldl"][1])

    heart_rate = 65 + 0.1 * bmi + np.random.normal(0, 10, n)
    heart_rate = np.clip(
        heart_rate, CLINICAL_RANGES["heart_rate"][0], CLINICAL_RANGES["heart_rate"][1])

    creatinine = np.where(
        df["sex"] == "Male",
        0.7 + 0.005 * age + np.random.exponential(0.15, n),
        0.5 + 0.004 * age + np.random.exponential(0.12, n)
    )
    creatinine = np.clip(
        creatinine, CLINICAL_RANGES["creatinine"][0], CLINICAL_RANGES["creatinine"][1])

    df["sbp"] = np.round(sbp, 0).astype(int)
    df["dbp"] = np.round(dbp, 0).astype(int)
    df["hba1c"] = np.round(hba1c, 2)
    df["ldl"] = np.round(ldl, 1)
    df["heart_rate"] = np.round(heart_rate, 0).astype(int)
    df["creatinine"] = np.round(creatinine, 2)

    return df

def generate_lifestyle(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    age = df["age"].values

    smoke_prob = 0.15 + 0.002 * (age - 40)
    smoke_prob = np.clip(smoke_prob, 0.05, 0.45)
    smoking_status = np.array([
        np.random.choice(
            ["Never", "Former", "Current"],
            p=[1 - p, p * 0.5, p * 0.5]
        )
        for p in smoke_prob
    ])

    bmi = df["weight_kg"] / (df["height_cm"] / 100) ** 2
    exercise_base = 4.0 - 0.02 * age - 0.03 * bmi + np.random.normal(0, 1.2, n)
    exercise_freq = np.clip(np.round(exercise_base), 0, 7).astype(int)

    df["smoking_status"] = smoking_status
    df["exercise_freq"] = exercise_freq

    return df

def assign_treatment(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    age = df["age"].values
    bmi = df["weight_kg"] / (df["height_cm"] / 100) ** 2

    risk_score = 0.3 * ((age - 40) / 30) + 0.3 *        ((bmi - 25) / 10) + 0.2 * (df["hba1c"] - 5.5) / 2
    risk_score = np.clip(risk_score, -1, 1)

    treatments = []
    for r in risk_score:
        if r > 0.3:
            p = [0.25, 0.55, 0.20]
        elif r < -0.3:
            p = [0.50, 0.20, 0.30]
        else:
            p = [0.40, 0.35, 0.25]
        treatments.append(np.random.choice(TREATMENT_ARMS, p=p))

    df["treatment_arm"] = treatments
    return df

def generate_outcome(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    bmi = df["weight_kg"] / (df["height_cm"] / 100) ** 2

    logit = (
        3.0
        - 0.25 * (df["age"] - 55)
        - 0.60 * (bmi - 26)
        - 3.0 * (df["hba1c"] - 5.5)
        - 0.15 * (df["sbp"] - 130)
        - 0.05 * (df["ldl"] - 100)
        + 1.2 * df["exercise_freq"]
        - 3.5 * (df["smoking_status"] == "Current").astype(float)
        - 1.5 * (df["smoking_status"] == "Former").astype(float)
        + 5.0 * (df["treatment_arm"] == "Enhanced").astype(float)
        + 1.5 * (df["treatment_arm"] == "Standard").astype(float)
        - 0.04 * df["heart_rate"]
        - 1.2 * (df["creatinine"] - 1.0)
        - 0.01 * (df["age"] - 55) * (df["hba1c"] - 5.5)
        - 0.005 * (df["sbp"] - 130) * (bmi - 26)
        + np.random.normal(0, 0.15, n)
    )

    prob = 1 / (1 + np.exp(-logit))

    current_rate = prob.mean()
    adjustment = np.log(TARGET_POSITIVE_RATE / (1 - TARGET_POSITIVE_RATE)
                        ) - np.log(current_rate / (1 - current_rate))
    prob_adjusted = 1 / (1 + np.exp(-(logit + adjustment)))

    df["outcome_binary"] = (np.random.random(n) < prob_adjusted).astype(int)
    df["_outcome_prob"] = np.round(prob_adjusted, 4)

    return df

def generate_costs(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    ca = COST_ASSUMPTIONS

    treatment_cost = np.where(
        df["treatment_arm"] == "Enhanced",
        np.random.normal(ca["treatment_cost_enhanced"], 1500, n),
        np.where(
            df["treatment_arm"] == "Standard",
            np.random.normal(ca["treatment_cost_standard"], 800, n),
            np.random.normal(ca["treatment_cost_control"], 300, n)
        )
    )
    treatment_cost = np.clip(treatment_cost, 500, 25000)

    downstream_cost = np.where(
        df["outcome_binary"] == 1,
        np.random.lognormal(np.log(ca["cost_good_outcome"]), 0.4, n),
        np.random.lognormal(np.log(ca["cost_bad_outcome"]), 0.5, n)
    )
    downstream_cost = np.clip(downstream_cost, 200, 150000)

    df["treatment_cost"] = np.round(treatment_cost, 2)
    df["downstream_cost"] = np.round(downstream_cost, 2)
    df["total_cost"] = np.round(treatment_cost + downstream_cost, 2)

    return df

def introduce_missingness(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rng = np.random.RandomState(RANDOM_SEED + 1)

    ldl_miss_prob = 0.15 - 0.002 * df["age"]
    ldl_miss_prob = np.clip(ldl_miss_prob, 0.02, 0.15)
    ldl_mask = rng.random(n) < ldl_miss_prob
    df.loc[ldl_mask, "ldl"] = np.nan

    ex_miss_prob = 0.02 + 0.001 * (df["age"] - 40)
    ex_miss_prob = np.clip(ex_miss_prob, 0.01, 0.10)
    ex_mask = rng.random(n) < ex_miss_prob
    df.loc[ex_mask, "exercise_freq"] = np.nan

    cr_mask = rng.random(n) < 0.03
    df.loc[cr_mask, "creatinine"] = np.nan

    return df

def validate_dataset(df: pd.DataFrame) -> None:
    logger.info(f"\n{'='*60}")
    logger.info("DATASET VALIDATION")
    logger.info(f"{'='*60}")
    logger.info(f"Shape: {df.shape}")
    logger.info(
        f"Outcome rate: {df['outcome_binary'].mean():.3f} (target: {TARGET_POSITIVE_RATE})")
    logger.info(f"\nOutcome by treatment arm:")
    logger.info(df.groupby("treatment_arm")["outcome_binary"].agg(["mean", "count"]))
    logger.info(f"\nMissingness:")
    missing = df.isnull().sum()
    logger.info(missing[missing > 0])
    logger.info(f"\nFeature ranges:")
    for col in ["age", "sbp", "dbp", "hba1c", "ldl", "heart_rate", "creatinine"]:
        if col in df.columns:
            logger.info(f"  {col}: [{df[col].min():.1f}, {df[col].max():.1f}]")
    logger.info(f"{'='*60}\n")

def main():
    logger.info("Generating synthetic healthcare dataset...")

    df = generate_demographics(N_SAMPLES)

    df = generate_clinical_markers(df)

    df = generate_lifestyle(df)

    df = assign_treatment(df)

    df = generate_outcome(df)

    df = generate_costs(df)

    df = introduce_missingness(df)

    df = df.drop(columns=["_outcome_prob"])

    validate_dataset(df)

    df.to_csv(DATA_FILE, index=False)
    logger.info(f"Dataset saved to {DATA_FILE}")
    logger.info(f"Columns: {list(df.columns)}")

    return df

if __name__ == "__main__":
    main()

