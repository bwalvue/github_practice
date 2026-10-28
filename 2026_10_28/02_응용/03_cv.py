"""실습 03 응용: train 내부 5-fold에서 Pipeline의 다중 지표 계산"""
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
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score
model = Pipeline([("pre", pre), ("model", LogisticRegression(C=1, max_iter=1000))])
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# TODO A03: train만 넣어 F1과 AUC를 함께 구하세요. 전처리한 배열 대신 model 전체를 전달합니다.
# Pipeline 전체 전달로 각 fold의 학습 48건에서만 전처리 학습
# test_f1·test_auc는 CV 내부 평가 12건의 점수이며 보관 test20과 무관
scores = cross_validate(
    model, X_train, y_train, cv=folds,
    scoring={"f1": make_scorer(f1_score, zero_division=0), "auc": "roc_auc"},
    n_jobs=1,
)
# TODO 끝
result = pd.DataFrame({"fold": range(1, 6), "F1": scores["test_f1"], "ROC_AUC": scores["test_auc"]})
result.to_csv(OUT / "classification_cv.csv", index=False, encoding="utf-8-sig")
print(result.round(3).to_string(index=False))
print("평균F1/표준편차:", round(result["F1"].mean(), 3), round(result["F1"].std(ddof=0), 3))
print("각 fold: 학습48·평가12. test_f1의 test는 CV 내부 평가이며 보관 test20이 아닙니다.")
