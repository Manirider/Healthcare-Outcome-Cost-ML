import os
import random
import numpy as np
RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUTS_DIR, "tables")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")

DATA_FILE = os.path.join(RAW_DATA_DIR, "synthetic_healthcare.csv")
ENGINEERED_DATA_FILE = os.path.join(
    PROCESSED_DATA_DIR, "synthetic_healthcare_engineered.csv")

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

N_SAMPLES = 8000
TARGET_POSITIVE_RATE = 0.33
TREATMENT_ARMS = ["Standard", "Enhanced", "Control"]

CLINICAL_RANGES = {
    "age": (18, 100),
    "height_cm": (120, 220),
    "weight_kg": (30, 250),
    "sbp": (70, 250),
    "dbp": (40, 150),
    "hba1c": (3.5, 15.0),
    "ldl": (30, 400),
    "heart_rate": (35, 200),
    "creatinine": (0.3, 12.0),
}

NUMERIC_FEATURES = [
    "age", "height_cm", "weight_kg", "sbp", "dbp",
    "hba1c", "ldl", "heart_rate", "creatinine", "exercise_freq"
]
CATEGORICAL_FEATURES = [
    "sex", "smoking_status", "treatment_arm"
]

TEST_SIZE = 0.20
CV_FOLDS = 5
CV_REPEATS = 3
OPTUNA_TRIALS = 120
CLASSIFICATION_THRESHOLD = 0.5

COST_ASSUMPTIONS = {
    "treatment_cost_standard": 5_000,
    "treatment_cost_enhanced": 12_000,
    "treatment_cost_control": 1_500,
    "cost_good_outcome": 2_000,
    "cost_bad_outcome": 35_000,
    "qaly_good_outcome": 0.85,
    "qaly_bad_outcome": 0.45,
    "discount_rate": 0.03,
    "time_horizon_years": 5,
}

WTP_THRESHOLDS = [20_000, 50_000, 100_000, 150_000]

AGE_BINS = [(18, 39, "Young"), (40, 59, "Middle-aged"), (60, 100, "Elderly")]
BMI_CATEGORIES = {
    "Underweight": (0, 18.5),
    "Normal": (18.5, 25),
    "Overweight": (25, 30),
    "Obese": (30, 100),
}

METABOLIC_SYNDROME_THRESHOLDS = {
    "sbp": 130,
    "dbp": 85,
    "hba1c": 5.7,
    "ldl": 130,
    "bmi": 30,
}

