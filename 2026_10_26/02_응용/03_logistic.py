"""실패 확률을 예측하고 혼동행렬에서 오탐과 놓침을 확인합니다."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
split = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = split.loc[split["split"] == "train"].sort_values("position")["request_id"]
valid_ids = split.loc[split["split"] == "valid"].sort_values("position")["request_id"]
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]

# failed나 실행 후에 생기는 status를 입력에 넣으면, 예측 시점에 없는 정답 정보를 학습에 사용하게 됩니다.
features = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train = train[features]
X_valid = valid[features]

# LogisticRegression은 이 CSV의 NaN을 그대로 받을 수 없으므로, 숫자·문자별 결측 대체를 모델 앞에 둡니다.
num = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
pre = ColumnTransformer([("num", num, features[:2]), ("cat", cat, features[2:])])

model = Pipeline([("pre", pre), ("model", LogisticRegression(C=1.0, max_iter=1000))])

# TODO A03: 실패 여부로 학습하고 X_valid를 예측하세요.
# 풀이: 전체 Pipeline을 fit해야 train으로 결측 대체값·변환 기준과 분류 모델이 함께 학습됩니다.
# predict는 그 기준을 그대로 X_valid에 적용하므로 전처리를 따로 다시 학습할 필요가 없습니다.
model.fit(X_train, train["failed"])
prediction = model.predict(X_valid)

print("클래스 순서:", model.classes_)

# [:, 1]은 '라벨 1'이 아니라 '두 번째 열' 선택입니다. 위의 classes_가 [0, 1]이라 실패 확률에 해당합니다.
print("앞 4건 실패 확률:", model.predict_proba(X_valid)[:4, 1].round(3))
print("혼동행렬 / 행=실제, 열=예측, 순서=[0,1]")

# 실제값을 첫 인자로 주어 행=실제·열=예측으로 고정합니다. 두 인자를 바꾸면 FP와 FN 해석이 뒤바뀝니다.
print(confusion_matrix(valid["failed"], prediction, labels=[0, 1]))
print("valid F1:", round(f1_score(valid["failed"], prediction, zero_division=0), 3))
