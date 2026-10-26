"""실습 02 심화: 기존 응용 수준의 회귀 비교와 오류 사례입니다."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
split = pd.read_csv(HERE / "data/split_manifest.csv")

# position으로 분할 내부 순서를 재현하고, request_id 인덱스의 .loc로 해당 요청을 선택합니다.
train_ids = split.loc[split["split"] == "train"].sort_values("position")["request_id"]
valid_ids = split.loc[split["split"] == "valid"].sort_values("position")["request_id"]
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]
features = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train = train[features]
X_valid = valid[features]

from preprocessing import build_model
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

rows = []
linear_prediction = None
for name, estimator in [("Dummy", DummyRegressor()), ("Linear", LinearRegression())]:
    model = build_model(estimator)

    # TODO H02: 회귀 모델을 학습하고 valid 예측을 구하세요.
    # 풀이: build_model은 전처리까지 포함하므로 Pipeline 전체를 fit해 대체값·스케일과 회귀 모델을 함께 학습합니다.
    model.fit(X_train, train["elapsed_ms"])
    prediction = model.predict(X_valid)
    rows.append({"model": name, "MAE_ms": mean_absolute_error(valid["elapsed_ms"], prediction),
                 "RMSE_ms": root_mean_squared_error(valid["elapsed_ms"], prediction),
                 "R2": r2_score(valid["elapsed_ms"], prediction)})

    # 모델 등록 순서에 기대지 않고 Linear의 예측만 보관해, 아래 잔차 분석의 대상을 고정합니다.
    if name == "Linear":
        linear_prediction = prediction
errors = valid[["elapsed_ms"]].copy()
errors["predicted_ms"] = linear_prediction

# 실제−예측이므로 과대 예측은 음수입니다. 큰 오차를 고를 때는 부호가 아닌 절댓값으로 정렬합니다.
errors["residual_ms"] = errors["elapsed_ms"] - errors["predicted_ms"]
errors["absolute_error_ms"] = errors["residual_ms"].abs()
worst = errors.sort_values("absolute_error_ms", ascending=False).head(3)
comparison = pd.DataFrame(rows)
out = HERE / "outputs"
out.mkdir(exist_ok=True)

comparison.to_csv(out / "regression_comparison.csv", index=False, encoding="utf-8-sig")
worst.to_csv(out / "largest_errors.csv", encoding="utf-8-sig")
print(comparison.round(3).to_string(index=False))
print("Linear의 큰 잔차 3건")
print(worst.round(3).to_string())
