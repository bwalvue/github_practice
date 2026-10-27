"""실습 03: 여러 트리를 결합하는 Random Forest를 실행합니다."""

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# 제공: 작은 숲 20개, 깊이 3. 오늘은 설정을 더 늘리지 않습니다.
model = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42, n_jobs=1)
# TODO B03: train으로 학습하고 valid를 예측하세요.
# 풀이: train 12건으로 숲을 학습하고 valid 4건의 라벨을 예측합니다.
# y_valid는 학습에 넣지 않고 아래 F1 계산에서 정답과 비교할 때 사용합니다.
model.fit(X_train, y_train)
prediction = model.predict(X_valid)
# estimators_는 fit 뒤 생성된 개별 트리 목록입니다.
# F1=0도 계산 결과일 수 있으므로 예측값을 직접 바꿔 맞추지 않습니다.

print("트리 수:", len(model.estimators_))
print("valid 예측:", prediction.tolist())
print("valid F1:", round(f1_score(y_valid, prediction, zero_division=0), 3))
