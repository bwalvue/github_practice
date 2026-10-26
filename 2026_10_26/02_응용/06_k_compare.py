"""전처리와 분할을 고정하고 다섯 K 후보의 학습·검증 F1을 비교합니다."""
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
for k in [1, 3, 5, 9, 15]:

    # TODO A06: n_neighbors를 반복 변수 k로 바꾸세요.
    # 풀이: 표의 K만 바꾸면 같은 모델의 점수를 다섯 번 기록하게 되므로 n_neighbors에도 k를 넣습니다.
    # clone(pre)로 전처리 객체까지 후보마다 분리해, 이전 반복에서 학습한 객체를 공유하지 않게 합니다.
    model = Pipeline([
        ("pre", clone(pre)),
        ("model", KNeighborsClassifier(n_neighbors=k)),
    ])
    model.fit(X_train, train["failed"])
    rows.append({
        "K": k,
        "train_F1": f1_score(train["failed"], model.predict(X_train), zero_division=0),
        "valid_F1": f1_score(valid["failed"], model.predict(X_valid), zero_division=0),
    })

comparison = pd.DataFrame(rows)

# K=1은 학습 행 자신이 거리 0인 이웃이라 train F1이 1이 됩니다. 선택 기준에는 valid F1을 사용합니다.
# 반올림 전 F1을 내림차순으로 비교하고, 동점일 때만 작은 K를 우선하도록 두 번째 정렬 기준을 둡니다.
best = comparison.sort_values(["valid_F1", "K"], ascending=[False, True]).iloc[0]
out = HERE / "outputs"
out.mkdir(exist_ok=True)
comparison.to_csv(out / "knn_search.csv", index=False, encoding="utf-8-sig")
print(comparison.round(3).to_string(index=False))
print("선택 K:", int(best["K"]))
print("저장: outputs/knn_search.csv")
