"""실습 04 응용: 같은 Logistic·5-fold에서 파생 특성 두 열의 효과 비교"""
from pathlib import Path
import pandas as pd
import numpy as np
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"].tolist()
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"].tolist()
train, valid = raw.loc[train_ids].copy(), raw.loc[valid_ids].copy()
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
NUM, CAT = FEATURES[:2], FEATURES[2:]
X_train, X_valid = train[FEATURES].copy(), valid[FEATURES].copy()
y_train, y_valid = train["failed"], valid["failed"]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# 제공: 평균·중앙값·범주 목록은 Pipeline의 학습 자료에서만 결정
pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                      ("scaler", StandardScaler())]), NUM),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
from sklearn.base import clone
# TODO A04: train·valid에 log_input_chars와 chars_per_slot을 추가하세요. 자기 행만 사용하고 NaN은 다음 Pipeline에서 처리합니다.
# 두 공식 모두 자기 행의 입력만 사용, 결측값은 NaN으로 유지
# 통계 누수 방지를 위해 중앙값·표준화는 fold별 Pipeline에서 학습
for part in (X_train, X_valid):
    part["log_input_chars"] = np.log1p(part["input_chars"])
    part["chars_per_slot"] = part["input_chars"] / (1.0 + part["allowed_tools"])
# TODO 끝
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))
choices = {"Logistic": NUM, "Logistic_FE": NUM + ["log_input_chars", "chars_per_slot"]}
rows = []
for name, numeric_columns in choices.items():
    current_pre = clone(pre)
    # 제공: 파생 열은 수치 경로에만 추가, 범주 경로 유지
    current_pre.transformers[0] = ("num", current_pre.transformers[0][1], numeric_columns)
    model = Pipeline([("pre", current_pre), ("model", LogisticRegression(C=1, max_iter=1000))])
    scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1", n_jobs=1)
    model.fit(X_train, y_train)
    rows.append({"model": name, "CV_F1": scores.mean(), "CV_std": scores.std(),
                 "valid_F1": f1_score(y_valid, model.predict(X_valid), zero_division=0)})
result = pd.DataFrame(rows)
result.to_csv(OUT / "feature_comparison.csv", index=False, encoding="utf-8-sig")
print(result.round(4).to_string(index=False))
print("CV는 선택용, valid는 진단용입니다. test는 아직 보지 않습니다.")
