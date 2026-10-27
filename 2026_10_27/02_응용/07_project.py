"""실습 07~08 응용: 얕은 폴더 구조에서 분류 실험표를 완성합니다."""

# AI 작성 참고 풀이: 원본 TODO를 유지하고 실행 코드와 해설을 추가했습니다.

from pathlib import Path
import pandas as pd

# 제공: 전날의 분할 소속뿐 아니라 position 순서까지 그대로 읽습니다.
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
# [분할표 해설] manifest는 각 요청 ID의 train/valid/test 소속과 그 안의 position을 정한 표입니다.
# position으로 ID 목록을 정렬한 뒤 raw.loc[train_ids]로 읽어야 전날과 같은 요청 순서가 됩니다.
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"]
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train, X_valid = train[FEATURES], valid[FEATURES]
y_train, y_valid = train["failed"], valid["failed"]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
# test 20건의 입력·정답은 아래 모델에 전달하지 않습니다.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# 제공: 전날 배운 결측 대체와 문자 변환입니다. 통계는 fit할 때만 배웁니다.
# [전처리 해설] 앞의 숫자 2열은 중앙값으로 결측을 채우고, 뒤의 문자 2열은 최빈값 대체 후 원핫 인코딩합니다.
# 이 데이터의 문자 열은 각각 3개 범주로 나뉘어, 모델 입력은 숫자 2열 + 원핫 6열 = 총 8열이 됩니다.
# clone(pre)는 설정이 같은 새 전처리기를 만듭니다. 후보마다 학습한 중앙값·범주 상태를 따로 가지게 합니다.
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), FEATURES[:2]),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
])
# 트리는 이 실습에서 표준화하지 않습니다. 각 후보·fold마다 새 전처리기를 학습합니다.

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, confusion_matrix

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))
candidates = [("Dummy", DummyClassifier(strategy="most_frequent")),
              ("Tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)), ("Forest", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
                                    max_features="sqrt", random_state=42, n_jobs=1)), ("XGB", XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", device="cpu", random_state=42, n_jobs=1))]
rows, cv_rows = [], []
predictions = pd.DataFrame({"request_id": X_valid.index, "actual": y_valid.to_numpy()})
for name, estimator in candidates:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO A07a: train 내부 교차검증 뒤 전체 train으로 학습하세요.
    # 풀이: Dummy·Tree·Forest·XGB를 같은 train 내부 5-fold로 비교합니다.
    scores = cross_val_score(
        model, X_train, y_train,
        cv=folds, scoring="f1",
    )
    # CV는 복제 모델을 학습하므로, valid 예측용 현재 Pipeline은 train 60건으로 다시 학습합니다.
    model.fit(X_train, y_train)
    # 후보 선택 기준은 CV_F1 최대이며 동점이면 candidates의 앞선 후보를 택합니다.
    # valid_F1·FP·FN은 별도 진단에 사용하고 test는 남겨 둡니다.

    # TODO A07b: valid의 예측을 prediction에 넣으세요.
    # 풀이: 학습한 Pipeline에 valid의 원본 입력 4열만 전달합니다.
    prediction = model.predict(X_valid)
    # 라벨 20개는 X_valid의 행 순서와 같아 request_id·actual과 요청별로 대조할 수 있습니다.
    # 정답 y_valid는 predict에 넣지 않고 아래 F1·혼동행렬 계산에서 비교합니다.

    tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()
    rows.append({"model": name, "CV_F1": scores.mean(), "CV_std": scores.std(),
                 "train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
                 "valid_F1": f1_score(y_valid, prediction, zero_division=0), "FP": fp, "FN": fn})
    for number, value in enumerate(scores, start=1):
        cv_rows.append({"model": name, "fold": number, "fit_n": 48, "evaluate_n": 12, "F1": value})
    predictions[name] = prediction
    if name == "Forest":
        importance = pd.DataFrame({"feature": model["pre"].get_feature_names_out(),
                                   "importance": model["model"].feature_importances_})
# [결과 읽기] comparison은 4후보의 요약 4행, cv_scores는 4후보 × 5-fold = 20행입니다.
# validation_predictions는 valid 요청 20행이며, feature_importance는 Forest의 변환 특성 8행입니다.
# CV_F1은 train 내부 평균, valid_F1·FP·FN은 바깥 valid 결과이므로 각각 해당 열에서 비교합니다.
comparison = pd.DataFrame(rows)
comparison.to_csv(OUT / "classification_comparison.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(cv_rows).to_csv(OUT / "cv_scores.csv", index=False, encoding="utf-8-sig")
predictions.to_csv(OUT / "validation_predictions.csv", index=False, encoding="utf-8-sig")
importance.to_csv(OUT / "feature_importance.csv", index=False, encoding="utf-8-sig")
print(comparison.round(4).to_string(index=False))
print("CV 평균 F1 최대, 동률이면 미리 제시한 후보 순서로 선택합니다. valid는 별도 진단입니다.")
print("저장: outputs/classification_comparison.csv, cv_scores.csv, validation_predictions.csv, feature_importance.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")
