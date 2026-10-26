"""같은 valid 20건에서 기준선과 선형회귀의 MAE·RMSE를 비교합니다."""
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
split = pd.read_csv(HERE / "data/split_manifest.csv")

# CSV에 적힌 행 순서와 분할 내부 순서는 다를 수 있어 position으로 정렬한 ID를 사용합니다.
# raw의 인덱스를 request_id로 설정했으므로 .loc[train_ids]는 행 번호가 아니라 ID로 선택합니다.
train_ids = split.loc[split["split"] == "train"].sort_values("position")["request_id"]
valid_ids = split.loc[split["split"] == "valid"].sort_values("position")["request_id"]
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]
features = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train = train[features]
X_valid = valid[features]

# 이 CSV에는 입력 결측이 있으므로 모델에 주기 전에 채웁니다.
# 숫자는 중앙값을 쓸 수 있지만 문자열은 중앙값을 구할 수 없어, 문자 경로에는 최빈값 대체를 둡니다.
num = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
pre = ColumnTransformer([("num", num, features[:2]), ("cat", cat, features[2:])])

rows = []

# 학습 정답의 평균만 내는 Dummy를 기준선으로 두면, 입력을 사용한 Linear가 실제로 오차를 줄였는지 비교할 수 있습니다.
for name, estimator in [("Dummy", DummyRegressor()), ("Linear", LinearRegression())]:

    # 후보마다 새 전처리 객체를 넣어, 한 후보의 fit이 다른 파이프라인의 학습 상태를 바꾸지 않게 합니다.
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])

    # TODO A02: 회귀 정답으로 학습하고 같은 valid를 예측하세요.
    # 풀이: X_train은 입력 4열이고 y는 실행시간 한 열입니다. predict에도 같은 4열 순서의 X_valid를 줍니다.
    model.fit(X_train, train["elapsed_ms"])
    prediction = model.predict(X_valid)

    # 같은 예측에 MAE와 RMSE를 모두 계산해야 평균적인 오차와 큰 오차의 영향을 따로 비교할 수 있습니다.
    rows.append({
        "model": name,
        "MAE_ms": mean_absolute_error(valid["elapsed_ms"], prediction),
        "RMSE_ms": root_mean_squared_error(valid["elapsed_ms"], prediction),
    })

comparison = pd.DataFrame(rows)
out = HERE / "outputs"
out.mkdir(exist_ok=True)
comparison.to_csv(out / "regression_comparison.csv", index=False, encoding="utf-8-sig")
print(comparison.round(3).to_string(index=False))
print("저장: outputs/regression_comparison.csv")
