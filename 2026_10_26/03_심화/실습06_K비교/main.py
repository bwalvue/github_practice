"""실습 06 심화: 기존 응용 수준의 분류 F1·회귀 RMSE 탐색입니다."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
split = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = split.loc[split["split"] == "train"].sort_values("position")["request_id"]
valid_ids = split.loc[split["split"] == "valid"].sort_values("position")["request_id"]
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]
features = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train = train[features]
X_valid = valid[features]

from preprocessing import build_model
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import f1_score, root_mean_squared_error

rows = []
for task in ["classification", "regression"]:

    # task에 맞춰 정답 열과 모델 종류를 함께 바꿔야 시간 회귀와 실패 분류가 섞이지 않습니다.
    target = "failed" if task == "classification" else "elapsed_ms"
    for k in [1, 3, 5, 9, 15]:
        estimator = KNeighborsClassifier(n_neighbors=k) if task == "classification" else KNeighborsRegressor(n_neighbors=k)
        model = build_model(estimator)
        model.fit(X_train, train[target])
        p_train = model.predict(X_train)
        p_valid = model.predict(X_valid)
        if task == "classification":
            train_score = f1_score(train[target], p_train, zero_division=0)
            valid_score = f1_score(valid[target], p_valid, zero_division=0)
            metric = "F1"
        else:

            # TODO H06: 회귀의 train·valid RMSE를 각각 구하세요.
            # 풀이: p_train은 train의 실제값, p_valid는 valid의 실제값과 짝지어 각각 RMSE를 구합니다.
            train_score = root_mean_squared_error(train[target], p_train)
            valid_score = root_mean_squared_error(valid[target], p_valid)
            metric = "RMSE_ms"
        rows.append({"task": task, "K": k, "scale": "standard", "weights": "uniform", "metric": metric,
                     "train_score": train_score, "valid_score": valid_score})
comparison = pd.DataFrame(rows)

# 같은 valid_score 열이라도 F1은 최대, RMSE는 최소를 골라야 하므로 정렬 방향을 나눕니다.
# 두 문제 모두 점수가 같을 때만 두 번째 기준인 작은 K를 우선합니다.
best_c = comparison[comparison["task"] == "classification"].sort_values(["valid_score", "K"], ascending=[False, True]).iloc[0]
best_r = comparison[comparison["task"] == "regression"].sort_values(["valid_score", "K"]).iloc[0]
out = HERE / "outputs"
out.mkdir(exist_ok=True)

comparison.to_csv(out / "knn_search.csv", index=False, encoding="utf-8-sig")
print(comparison.round(3).to_string(index=False))
print("선택 / 분류 K:", int(best_c["K"]), "회귀 K:", int(best_r["K"]))
