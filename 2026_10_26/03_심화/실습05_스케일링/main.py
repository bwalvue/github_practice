"""실습 05 심화: 예측 전후 학습 통계를 함께 관찰합니다."""
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

import numpy as np
from preprocessing import build_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

rows = []
for scale in ["none", "standard", "minmax"]:

    # TODO H05: 전처리에 반복 변수 scale을 전달하세요.
    # 풀이: standard를 고정하면 표의 이름만 none·minmax로 바뀝니다. scale을 함수에 전달해야 실제 전처리도 바뀝니다.
    model = build_model(KNeighborsClassifier(n_neighbors=5), scale=scale)
    model.fit(X_train, train["failed"])
    imputer = model["pre"].named_transformers_["num"]["imputer"]

    # 같은 배열의 참조를 보관하면 제자리 변경 시 비교 대상도 바뀝니다. copy로 예측 전 중앙값을 따로 보관합니다.
    before = imputer.statistics_.copy()
    prediction = model.predict(X_valid)
    rows.append({"scale": scale, "K": 5,
        "train_F1": f1_score(train["failed"], model.predict(X_train), zero_division=0),
        "valid_F1": f1_score(valid["failed"], prediction, zero_division=0),
        "median_unchanged": np.array_equal(before, imputer.statistics_)})
comparison = pd.DataFrame(rows)
out = HERE / "outputs"
out.mkdir(exist_ok=True)

comparison.to_csv(out / "scaling_comparison.csv", index=False, encoding="utf-8-sig")
print(comparison.round(3).to_string(index=False))
print("숫자와 문자 처리를 유지하고, 숫자의 스케일만 바꾸었습니다.")
