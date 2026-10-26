"""실습 07: 회귀·분류 비교표와 모델을 생성합니다. 실습08은 restore.py로 이어갑니다."""
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

import json
import joblib
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from preprocessing import build_model
from evaluate import regression_metrics, classification_metrics
from search import compare_k

# K를 정답 값으로 고정하지 않고 실제 탐색표에서 고릅니다. 분류는 F1 내림차순, 회귀는 RMSE 오름차순입니다.
search = compare_k(train, valid)
best_c = search[search["task"] == "classification"].sort_values(["valid_score", "K"], ascending=[False, True]).iloc[0]
best_r = search[search["task"] == "regression"].sort_values(["valid_score", "K"]).iloc[0]
k_class, k_reg = int(best_c["K"]), int(best_r["K"])
specifications = {
    "regression": [("Dummy", DummyRegressor()), ("Linear", LinearRegression()),
                   (f"KNN{k_reg}", KNeighborsRegressor(n_neighbors=k_reg))],
    "classification": [("Dummy", DummyClassifier(strategy="most_frequent")),
                       ("Logistic", LogisticRegression(C=1.0, max_iter=1000)),
                       (f"KNN{k_class}", KNeighborsClassifier(n_neighbors=k_class))],
}
out = HERE / "outputs"
out.mkdir(exist_ok=True)

predictions = []
selection = {}
chosen_models = {}
for task, candidates in specifications.items():
    target = "elapsed_ms" if task == "regression" else "failed"
    rows = []

    # 선택 후 새 모델을 만들면 학습 상태가 없으므로, 학습된 Pipeline을 이름별로 보관해 그 객체를 저장합니다.
    fitted = {}
    for name, estimator in candidates:
        model = build_model(estimator)

        # TODO H07fit: 해당 target으로 학습하고 valid 예측을 구하세요.
        # 풀이: 현재 task가 정한 target으로 학습합니다. 같은 입력 4열에서도 회귀는 실행시간, 분류는 실패 라벨을 정답으로 씁니다.
        model.fit(X_train, train[target])
        prediction = model.predict(X_valid)
        metric = regression_metrics(valid[target], prediction) if task == "regression" else classification_metrics(valid[target], prediction)
        rows.append({"model": name, "scale": "standard", **metric})

        # 한 요청에 모델별·문제별 예측이 쌓이므로 request_id 외에 task와 model도 저장해 행을 구분합니다.
        predictions.append(pd.DataFrame({"request_id": valid.index, "task": task, "model": name,
            "actual": valid[target].to_numpy(), "predicted": prediction}))
        fitted[name] = model
    table = pd.DataFrame(rows)
    table.to_csv(out / f"{task}_comparison.csv", index=False, encoding="utf-8-sig")

    # 정확도만 보면 성공 14건을 찍는 Dummy도 0.70이므로 분류는 실패 1의 F1로 선택합니다.
    # idxmin/idxmax는 첫 최솟값/최댓값의 행을 반환해 동점이면 등록 순서의 단순 후보를 유지합니다.
    best = table.loc[table["RMSE_ms"].idxmin()] if task == "regression" else table.loc[table["F1"].idxmax()]
    selection[task] = best["model"]
    chosen_models[task] = fitted[best["model"]]
    print(task, "/ valid 20건")
    print(table.round(3).to_string(index=False))

search.to_csv(out / "knn_search.csv", index=False, encoding="utf-8-sig")
pd.concat(predictions, ignore_index=True).to_csv(out / "validation_predictions.csv", index=False, encoding="utf-8-sig")
(out / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")

# 저장 대상은 추정기만이 아니라 학습된 전처리를 포함한 Pipeline입니다. 평균·중앙값·범주 대응까지 함께 복원합니다.
joblib.dump(chosen_models, out / "models.joblib")
print("선택 모델:", selection)
print("선택 K / 분류:", k_class, "회귀:", k_reg)
print("outputs에 비교표·탐색표·예측·선택 기록·모델을 저장했습니다.")
print("test는 보관만 했으며 모델에 전달하지 않았습니다.")
