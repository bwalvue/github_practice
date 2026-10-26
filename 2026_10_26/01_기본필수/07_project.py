"""같은 검증 자료에서 실행시간 회귀와 실패 여부 분류를 각각 비교합니다."""
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "requests.csv")
develop, test = train_test_split(
    raw, test_size=0.2, random_state=42, stratify=raw["failed"]
)
train, valid = train_test_split(
    develop, test_size=0.25, random_state=42, stratify=develop["failed"]
)

# 예측 대상인 elapsed_ms·failed를 입력에 넣으면 정답을 보고 맞히게 되므로 features에서 제외합니다.
features = ["input_chars", "task_type"]
X_train = train[features]
X_valid = valid[features]

# 선형 모델에 작업 종류 문자열을 그대로 넣을 수 없어 원핫으로 바꾸고, 숫자 열과 처리 경로를 나눕니다.
pre = ColumnTransformer([
    ("num", StandardScaler(), ["input_chars"]),
    ("cat", OneHotEncoder(sparse_output=False), ["task_type"]),
])

regression_rows = []
classification_rows = []
result = valid[["request_id", "elapsed_ms", "failed"]].copy()

for name, estimator in [
    ("Dummy", DummyRegressor()),
    ("Linear", LinearRegression()),
]:

    # pre 하나를 공유하면 한 후보의 fit이 다른 후보의 전처리 객체도 갱신하므로, clone으로 따로 만듭니다.
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])

    # TODO B07a: 실행시간 elapsed_ms로 학습하고 X_valid를 예측하세요.
    # 풀이: 회귀의 정답은 실행시간입니다. fit에 elapsed_ms를 주고, 같은 입력 열 순서의 X_valid를 예측합니다.
    model.fit(X_train, train["elapsed_ms"])
    prediction = model.predict(X_valid)

    regression_rows.append({
        "model": name,
        "MAE_ms": mean_absolute_error(valid["elapsed_ms"], prediction),
    })
    result[name + "_ms"] = prediction

for name, estimator in [
    ("Dummy", DummyClassifier(strategy="most_frequent")),
    ("Logistic", LogisticRegression(max_iter=1000)),
]:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])

    # TODO B07b: 실패 여부 failed로 학습하고 X_valid를 예측하세요.
    # 풀이: 입력은 회귀와 같지만, 분류에는 성공/실패 라벨인 failed를 정답으로 전달해야 합니다.
    model.fit(X_train, train["failed"])
    prediction = model.predict(X_valid)

    classification_rows.append({
        "model": name,
        "F1": f1_score(valid["failed"], prediction, zero_division=0),
    })
    result[name + "_failed"] = prediction

regression = pd.DataFrame(regression_rows)
classification = pd.DataFrame(classification_rows)
regression.to_csv(HERE / "regression_comparison.csv", index=False, encoding="utf-8-sig")
classification.to_csv(
    HERE / "classification_comparison.csv", index=False, encoding="utf-8-sig"
)
result.to_csv(HERE / "validation_predictions.csv", index=False, encoding="utf-8-sig")

print("분할:", len(train), len(valid), len(test))
print("회귀 비교 / 낮은 MAE가 좋음")
print(regression.round(3).to_string(index=False))
print("분류 비교 / 높은 F1이 좋음")
print(classification.round(3).to_string(index=False))
print("저장: regression_comparison.csv, classification_comparison.csv, validation_predictions.csv")
print("test는 학습·예측하지 않았습니다.")
