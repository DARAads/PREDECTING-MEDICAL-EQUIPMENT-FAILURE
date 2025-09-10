import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report

from xgboost import XGBClassifier                                                                                                                     # type: ignore

# Load dataset
df = pd.read_csv("SynDataset.csv")

# Drop leakage columns
leakage_cols = ["PredictedStatus", "PredictedTimeToFailure", "PerformanceScore"]
df = df.drop(columns=[col for col in leakage_cols if col in df.columns], errors="ignore")

# Convert LastServiceDate to datetime and drop (optional)
if "LastServiceDate" in df.columns:
    df["LastServiceDate"] = pd.to_datetime(df["LastServiceDate"], format="%d-%m-%Y", errors="coerce")
    df = df.drop(columns=["LastServiceDate"])

# Add Gaussian noise to numeric columns except target
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
target = "FailureRisk"
if target in numeric_features:
    numeric_features.remove(target)

noise_level = 0.30  # Adjust noise to control accuracy
rng = np.random.default_rng(42)
for col in numeric_features:
    df[col] = df[col] + noise_level * rng.normal(loc=0, scale=df[col].std(), size=df.shape[0])

# Define features and target
X = df.drop(columns=[target])
y = df[target]

# Define categorical and numeric features
possible_categorical = ["DeviceType", "DeviceName", "ClimateControl", "Location"]
categorical_features = [col for col in possible_categorical if col in X.columns]

# Add other object dtype columns if any
for col in X.select_dtypes(exclude=[np.number]).columns.tolist():
    if col not in categorical_features:
        categorical_features.append(col)

categorical_features = list(dict.fromkeys(categorical_features))
numeric_features = [col for col in X.columns if col not in categorical_features]

# Define the categories for one-hot encoding based on the lists used in the publisher
DeviceType_list = ["Anesthesia Machine","CT Scanner","Defibrillator","Dialysis Machine",
                   "ECG Monitor","Infusion Pump","Patient Ventilator","Ultrasound Machine"]
DeviceName_list = ["Alaris GH","Baxter AK 96","Baxter Flo-Gard","Datex Ohmeda S5","Drager Fabius Trio",
                   "Drager V500","Fresenius 4008","GE Aisys","GE Logiq E9","GE MAC 2000","GE Revolution",
                   "Hamilton G5","HeartStart FRx","Lifepak 20","NxStage System One","Philips EPIQ",
                   "Philips HeartStrart","Philips Ingenuity","Phillips PageWriter","Puritan Bennett 980",
                   "Siemens Acuson","Siemens S2000","Smiths Medfusion","Zoll R Series"]
ClimateControl_list = ["Yes","No"]
Location_list = [
    "Hospital A - Central Region","Hospital A - East Region","Hospital A - North Region","Hospital A - South Region","Hospital A - West Region",
    "Hospital B - Central Region","Hospital B - East Region","Hospital B - North Region","Hospital B - South Region","Hospital B - West Region",
    "Hospital C - Central Region","Hospital C - East Region","Hospital C - North Region","Hospital C - South Region","Hospital C - West Region",
    "Hospital D - Central Region","Hospital D - East Region","Hospital D - North Region","Hospital D - South Region","Hospital D - West Region",
    "Hospital E - Central Region","Hospital E - East Region","Hospital E - North Region","Hospital E - South Region","Hospital E - West Region",
    "Hospital F - Central Region","Hospital F - East Region","Hospital F - North Region","Hospital F - South Region","Hospital F - West Region",
    "Hospital G - Central Region","Hospital G - East Region","Hospital G - North Region","Hospital G - South Region","Hospital G - West Region",
    "Hospital H - Central Region","Hospital H - East Region","Hospital H - North Region","Hospital H - South Region","Hospital H - West Region"
]

# Map categorical features to their defined categories
categorical_categories = {
    "DeviceType": DeviceType_list,
    "DeviceName": DeviceName_list,
    "ClimateControl": ClimateControl_list,
    "Location": Location_list
}

# Build preprocessing pipelines
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Create tuples for categories in the OneHotEncoder, handling potential missing features in the dataset
ohe_categories = [categorical_categories[feature] for feature in categorical_features if feature in categorical_categories]

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, categories=ohe_categories)),
])


preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Encode target
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

# Build pipeline with preprocessor and XGBoost model
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        n_estimators=40,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ))
])

# Train model
pipeline.fit(X_train, y_train)

# Predict on test set
y_pred = pipeline.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"✅ XGBoost - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))