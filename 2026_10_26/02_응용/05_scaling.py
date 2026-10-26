"""K=5와 분할을 고정한 채 숫자 열의 스케일링 방식만 비교합니다."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

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

# 글자 수는 203~2381, 도구 수는 1~5라 원단위 거리에서는 글자 수 차이가 훨씬 크게 작용합니다.
# 스케일링만 바꾸었을 때 가까운 이웃과 예측이 어떻게 달라지는지 비교합니다.
rows = []
for name, scaler in [
    ("none", "passthrough"),
    ("standard", StandardScaler()),
    ("minmax", MinMaxScaler()),
]:

    # passthrough에서도 결측 대체는 남겨 두어야, 후보 간 차이가 결측 처리 여부가 아닌 스케일링 방식에만 생깁니다.
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([("num", num, features[:2]), ("cat", cat, features[2:])])
    model = Pipeline([("pre", pre), ("model", KNeighborsClassifier(n_neighbors=5))])

    # TODO A05: train에서 학습하고 valid 예측을 구하세요.
    # 풀이: Pipeline.fit으로 train의 대체값·스케일을 배우고, predict에서 그 변환을 X_valid에 재사용합니다.
    # valid를 따로 fit_transform하면 train과 다른 기준으로 변환돼, 같은 좌표계에서 KNN 거리를 비교할 수 없습니다.
    model.fit(X_train, train["failed"])
    prediction = model.predict(X_valid)
    rows.append({
        "scale": name,
        "K": 5,
        "valid_F1": f1_score(valid["failed"], prediction, zero_division=0),
    })

comparison = pd.DataFrame(rows)
out = HERE / "outputs"
out.mkdir(exist_ok=True)
comparison.to_csv(out / "scaling_comparison.csv", index=False, encoding="utf-8-sig")
print(comparison.round(3).to_string(index=False))
print("저장: outputs/scaling_comparison.csv")
