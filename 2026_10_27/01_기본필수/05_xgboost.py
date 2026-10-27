"""실습 05: XGBoost도 같은 fit·predict 흐름으로 실행합니다."""

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

from xgboost import XGBClassifier
from sklearn.metrics import f1_score

# 제공: 학습 회차 80·깊이 2·보정 크기 0.05를 고정합니다. CPU로 실행합니다.
model = XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
                      objective="binary:logistic", eval_metric="logloss",
                      tree_method="hist", random_state=42, n_jobs=1)
# TODO B05: train으로 학습하고 valid를 예측하세요.
# 풀이: XGBoost도 train으로 fit하고 valid 입력만으로 predict합니다.
# 80회차·깊이 2·학습률 0.05 등 제공된 설정은 유지합니다.
model.fit(X_train, y_train)
prediction = model.predict(X_valid)
# binary:logistic은 이진 분류 목적이고 logloss는 설정된 평가 지표입니다.
# 아래 F1은 예측 라벨 0/1을 실제 정답과 비교해 별도로 계산합니다.

print("valid 예측:", prediction.tolist())
print("valid F1:", round(f1_score(y_valid, prediction, zero_division=0), 3))
print("학습 목적: 실패 분류 / 확인 지표: F1")
