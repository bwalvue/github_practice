"""실습 07~08: 분류 후보 2개를 비교하고 같은 파일을 다시 실행합니다."""

# AI 작성 참고 풀이: 원본 TODO를 유지하고 실행 코드와 해설을 추가했습니다.

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 제공: 이 파일 옆의 정상 CSV 20건을 읽습니다. 현재 터미널 위치와 무관합니다.
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "requests.csv")
# [분할 해설]
# 첫 분할은 20건 중 20%인 4건을 test로 분리하고, 남은 develop은 16건입니다.
# 둘째 분할은 그 16건의 25%인 4건을 valid로 분리하므로 최종 train은 12건입니다.
# test_size는 두 번째 묶음의 비율을 뜻하는 인자 이름이라 valid를 나눌 때도 사용합니다.
# stratify는 0/1 비율을 가능한 한 유지하고, random_state=42는 같은 분할을 재현합니다.
develop, test = train_test_split(raw, test_size=0.2, stratify=raw["failed"], random_state=42)
train, valid = train_test_split(develop, test_size=0.25, stratify=develop["failed"], random_state=42)
# test 4건은 보관만 합니다. 입력은 실행 전에 아는 글자 수입니다.
# [입력 모양] [[...]]는 열 1개짜리 2차원 표를 유지합니다. train 입력은 (12, 1), valid는 (4, 1)입니다.
# 정답 y_train은 요청마다 하나인 0/1 값을 담는 1차원 Series입니다.
X_train = train[["input_chars"]]
X_valid = valid[["input_chars"]]
y_train = train["failed"]
y_valid = valid["failed"]

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import f1_score

# 제공: 전날과 같은 입력 2열입니다. 실행시간과 failed는 입력에 넣지 않습니다.
# [통합 입력] 이곳에서 앞서 만든 글자 수 1열 입력을 글자 수·작업 종류 2열 입력으로 다시 지정합니다.
# 숫자는 passthrough로 유지하고, task_type의 qa·summary는 원핫 2열로 바꾸므로 최종 모델 입력은 3열입니다.
features = ["input_chars", "task_type"]
X_train, X_valid = train[features], valid[features]
pre = ColumnTransformer([
    ("num", "passthrough", ["input_chars"]),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["task_type"]),
])
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
candidates = [("Tree", DecisionTreeClassifier(max_depth=3, random_state=42)),
              ("Forest", RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42, n_jobs=1))]
rows, cv_rows = [], []
predictions = valid[["request_id", "failed"]].copy()
for name, estimator in candidates:
    # 제공: 전처리까지 Pipeline에 포함하여 fold 학습 부분에서만 fit하게 합니다.
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO B07a: 전처리까지 연결한 model을 train에서 교차검증하세요.
    # 풀이: 문자 열이 남아 있는 원본 입력과 Pipeline 전체를 CV에 전달합니다.
    # 각 fold의 학습 8건으로 원핫 인코딩과 모델을 함께 학습해 정보 누수를 막습니다.
    scores = cross_val_score(
        model, X_train, y_train,
        cv=folds, scoring="f1",
    )
    # CV는 복제한 Pipeline들을 학습합니다. 현재 model의 전체 train 학습은 다음 TODO에서 합니다.

    # TODO B07b: 전체 train으로 학습하고 valid를 예측하세요.
    # 풀이: 현재 Pipeline을 train 12건 전체로 학습한 뒤 valid 4건을 예측합니다.
    model.fit(X_train, y_train)
    prediction = model.predict(X_valid)
    # CV_F1은 train 내부 평가의 평균, valid_F1은 별도 valid의 점수입니다.
    # test 4건은 두 작업에 모두 사용하지 않습니다.

    rows.append({"model": name, "CV_F1": scores.mean(),
                 "CV_std": scores.std(), "valid_F1": f1_score(y_valid, prediction, zero_division=0)})
    for number, value in enumerate(scores, start=1):
        cv_rows.append({"model": name, "fold": number, "F1": value})
    predictions[name] = prediction
    if name == "Forest":
        importance = pd.DataFrame({"feature": model["pre"].get_feature_names_out(),
                                   "importance": model["model"].feature_importances_})

# 제공: 입력 파일을 덮어쓰지 않고 결과 4개를 같은 폴더에 만듭니다.
# [결과 읽기] comparison은 후보 2행, cv_scores는 후보 2개 × fold 3개 = 6행입니다.
# validation_predictions는 valid 요청 4행, feature_importance는 Forest가 받은 변환 특성 3행입니다.
# model['pre']와 model['model']은 이름으로 Pipeline의 각 단계를 꺼내는 표기입니다.
comparison = pd.DataFrame(rows)
comparison.to_csv(HERE / "classification_comparison.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(cv_rows).to_csv(HERE / "cv_scores.csv", index=False, encoding="utf-8-sig")
predictions.to_csv(HERE / "validation_predictions.csv", index=False, encoding="utf-8-sig")
importance.to_csv(HERE / "feature_importance.csv", index=False, encoding="utf-8-sig")
print("분할:", len(train), len(valid), len(test))
print(comparison.round(3).to_string(index=False))
print("중요도")
print(importance.round(3).to_string(index=False))
print("저장: classification_comparison.csv, cv_scores.csv, validation_predictions.csv, feature_importance.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")
