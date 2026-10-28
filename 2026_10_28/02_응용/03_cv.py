"""실습 03 응용: Pipeline을 통째로 train 내부 5-fold에 넣어 다중 지표를 계산합니다."""
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
# 제공: 모든 평균·중앙값·범주 목록은 Pipeline.fit의 학습 부분에서만 결정됩니다.
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
# 정답 예시 A03 (아래 임시 대입문을 다음 호출로 교체):
# scores = cross_validate(
#     model, X_train, y_train, cv=folds,
#     scoring={"f1": make_scorer(f1_score, zero_division=0), "auc": "roc_auc"},
#     n_jobs=1,
# )
# 해설: cross_validate는 같은 5개 fold에서 여러 평가 지표를 함께 계산합니다.
# Pipeline 전체를 전달하므로 중앙값·표준화·범주 목록도 각 fold의 학습48건에서만 배웁니다.
# f1 scorer는 0/1 예측값을, roc_auc scorer는 실패1의 연속 점수를 사용하도록 구성됩니다.
# scoring의 키 f1·auc에 test_가 붙어 test_f1·test_auc 배열이 반환되며 길이는 각각 5입니다.
# 여기서 test는 fold 내부 평가12건을 뜻합니다. 보관한 test20이나 바깥 valid20을 넣지 않습니다.
# 점수의 표준편차는 fold 간 변동을 보여 주며 신뢰구간이나 최종 성능 보장은 아닙니다.
scores = {"test_f1": np.zeros(5), "test_auc": np.zeros(5)}
# TODO 끝
result = pd.DataFrame({"fold": range(1, 6), "F1": scores["test_f1"], "ROC_AUC": scores["test_auc"]})
result.to_csv(OUT / "classification_cv.csv", index=False, encoding="utf-8-sig")
print(result.round(3).to_string(index=False))
print("평균F1/표준편차:", round(result["F1"].mean(), 3), round(result["F1"].std(ddof=0), 3))
print("각 fold: 학습48·평가12. test_f1의 test는 CV 내부 평가이며 보관 test20이 아닙니다.")
