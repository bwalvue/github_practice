"""실습 02 응용: 기존 PPT의 5개 깊이 후보를 비교합니다."""

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
from sklearn.metrics import f1_score

rows = []
for depth in [1, 2, 3, 5, None]:
    # TODO A02: depth를 실제 트리 깊이로 적용하세요.
    # 풀이: 표에 기록하는 depth와 실제 트리의 max_depth를 일치시킵니다.
    # None은 깊이 상한을 두지 않되, 다른 분할 종료 조건은 유지한다는 뜻입니다.
    estimator = DecisionTreeClassifier(
        max_depth=depth, min_samples_leaf=1, random_state=42,
    )
    # 깊이 외의 조건은 고정합니다. 아래 get_n_leaves()로 실제 잎 수도 확인합니다.

    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    model.fit(X_train, y_train)
    rows.append({"depth": str(depth), "train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
                 "valid_F1": f1_score(y_valid, model.predict(X_valid), zero_division=0),
                 "leaves": model["model"].get_n_leaves()})
result = pd.DataFrame(rows)
result.to_csv(OUT / "02_depth.csv", index=False, encoding="utf-8-sig")
print(result.round(3).to_string(index=False))
