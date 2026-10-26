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

# elapsed_ms·failed는 맞혀야 할 정답이고 status·retry_count는 실행 후 정보라, 실행 전 입력 4열만 선택합니다.
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

# 문자열을 숫자 모델에 바로 넣을 수 없어 문자 경로는 원핫, 숫자 경로는 표준화를 적용한 뒤 합칩니다.
pre = ColumnTransformer([("num", num, features[:2]), ("cat", cat, features[2:])])

# K=3·K=15는 문제에서 제공한 값을 고정해 사용합니다. 이번 비교에서는 K 탐색 대신 모델 종류를 바꿉니다.
regression_rows = []
classification_rows = []

# 한 ID에 회귀·분류 각 3개 예측이 생기므로, 결과에는 ID와 함께 task·model을 기록해 서로 구분합니다.
predictions = []

for name, estimator in [
    ("Dummy", DummyRegressor()),
    ("Linear", LinearRegression()),
    ("KNN15", KNeighborsRegressor(n_neighbors=15)),
]:

    # pre를 그대로 공유하면 여러 후보가 같은 객체를 가리키므로, clone으로 학습 상태가 독립인 객체를 넣습니다.
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])

    # TODO A07r: 회귀 모델을 학습하고 valid 예측을 구하세요.
    # 풀이: 회귀는 실행시간을 예측하므로 y에 elapsed_ms를 주고, valid 정답은 예측 후 오차 계산에 사용합니다.
    model.fit(X_train, train["elapsed_ms"])
    prediction = model.predict(X_valid)

    regression_rows.append({
        "model": name,
        "scale": "standard",
        "MAE_ms": mean_absolute_error(valid["elapsed_ms"], prediction),
        "RMSE_ms": root_mean_squared_error(valid["elapsed_ms"], prediction),
    })
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

    # TODO A07c: 분류 모델을 학습하고 valid 예측을 구하세요.
    # 풀이: 같은 입력으로도 분류의 y는 성공/실패 라벨이어야 하므로, 회귀의 elapsed_ms 대신 failed를 사용합니다.
    model.fit(X_train, train["failed"])
    prediction = model.predict(X_valid)

    # labels=[0, 1]로 행·열 순서를 고정해야 펼친 네 값을 TN, FP, FN, TP로 일관되게 받을 수 있습니다.
    tn, fp, fn, tp = confusion_matrix(valid["failed"], prediction, labels=[0, 1]).ravel()

    # Dummy가 실패를 하나도 예측하지 않으면 Precision의 분모가 0이 되므로, zero_division=0으로 비교표에 0을 기록합니다.
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
pd.concat(predictions, ignore_index=True).to_csv(
    out / "validation_predictions.csv", index=False, encoding="utf-8-sig"
)

print("회귀 / valid 20건")
print(regression.round(3).to_string(index=False))
print("분류 / valid 20건")
print(classification.round(3).to_string(index=False))
print("저장: outputs의 비교표 2개와 validation_predictions.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")
