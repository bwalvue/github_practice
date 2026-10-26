"""회귀·분류 후보를 같은 분할에서 비교하고 요청별 검증 예측을 저장합니다."""
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
split = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = split.loc[split["split"] == "train"].sort_values("position")["request_id"]
valid_ids = split.loc[split["split"] == "valid"].sort_values("position")["request_id"]
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]

# 정답 elapsed_ms·failed와 실행 후 정보 status·retry_count는 입력에서 제외합니다.
features = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train = train[features]
X_valid = valid[features]
num = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
pre = ColumnTransformer([("num", num, features[:2]), ("cat", cat, features[2:])])

# 분류 K=3·회귀 K=15는 강의의 동일 조건 탐색에서 제공된 값을 재사용합니다.
# 이 파일에서 K 탐색을 새로 수행한 것은 아닙니다.
regression_rows = []
classification_rows = []
predictions = []

for name, estimator in [
    ("Dummy", DummyRegressor()),
    ("Linear", LinearRegression()),
    ("KNN15", KNeighborsRegressor(n_neighbors=15)),
]:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    model.fit(X_train, train["elapsed_ms"])
    prediction = model.predict(X_valid)

    regression_rows.append({
        "model": name,
        "scale": "standard",
        "MAE_ms": mean_absolute_error(valid["elapsed_ms"], prediction),
        "RMSE_ms": root_mean_squared_error(valid["elapsed_ms"], prediction),
    })
    # 후보별 실제값과 예측값을 같은 ID의 행에 기록하여 오차를 추적합니다.
    predictions.append(pd.DataFrame({
        "request_id": valid.index,
        "task": "regression",
        "model": name,
        "actual": valid["elapsed_ms"].to_numpy(),
        "predicted": prediction,
    }))

for name, estimator in [
    ("Dummy", DummyClassifier(strategy="most_frequent")),
    ("Logistic", LogisticRegression(C=1.0, max_iter=1000)),
    ("KNN3", KNeighborsClassifier(n_neighbors=3)),
]:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    model.fit(X_train, train["failed"])
    prediction = model.predict(X_valid)

    # ravel()은 2×2 표를 [TN, FP, FN, TP] 순서의 1차원 배열로 펼칩니다.
    tn, fp, fn, tp = confusion_matrix(valid["failed"], prediction, labels=[0, 1]).ravel()
    classification_rows.append({
        "model": name,
        "scale": "standard",
        "F1": f1_score(valid["failed"], prediction, zero_division=0),
        "Precision": precision_score(valid["failed"], prediction, zero_division=0),
        "Recall": recall_score(valid["failed"], prediction, zero_division=0),
        "FP": fp,
        "FN": fn,
    })
    predictions.append(pd.DataFrame({
        "request_id": valid.index,
        "task": "classification",
        "model": name,
        "actual": valid["failed"].to_numpy(),
        "predicted": prediction,
    }))

regression = pd.DataFrame(regression_rows)
classification = pd.DataFrame(classification_rows)
out = HERE / "outputs"
out.mkdir(exist_ok=True)
regression.to_csv(out / "regression_comparison.csv", index=False, encoding="utf-8-sig")
classification.to_csv(
    out / "classification_comparison.csv", index=False, encoding="utf-8-sig"
)
# 20건 × 3후보 × 2문제 = 120행입니다. index=False는 불필요한 행 번호 저장을 막습니다.
pd.concat(predictions, ignore_index=True).to_csv(
    out / "validation_predictions.csv", index=False, encoding="utf-8-sig"
)

print("회귀 / valid 20건")
print(regression.round(3).to_string(index=False))
print("분류 / valid 20건")
print(classification.round(3).to_string(index=False))
print("저장: outputs의 비교표 2개와 validation_predictions.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")

# 주 지표로 회귀는 RMSE 최소인 Linear, 분류는 F1 최대인 KNN3를 선택합니다.
# 회귀의 Linear와 KNN15는 RMSE 차이가 약 1.8ms라서 일반적인 우열까지 단정하기 어렵습니다.
# test는 모델에 전달하지 않았고, 합성 자료의 valid 비교만으로 실제 서비스 성능을 보장하지 않습니다.
