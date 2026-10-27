"""실습 06: train 12건 안에서만 3-fold 교차검증합니다."""

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
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 제공: 각 클래스 수를 고려해 기본필수는 3-fold입니다. valid4·test4는 넣지 않습니다.
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
model = DecisionTreeClassifier(max_depth=3, random_state=42)
# TODO B06: model을 train에서 3-fold F1으로 평가하세요.
# 풀이: 바깥 valid·test를 제외하고 train 12건만 3-fold로 평가합니다.
# 매번 8건으로 학습하고 남은 4건으로 평가하며, 실패 1의 F1을 구합니다.
scores = cross_val_score(
    model, X_train, y_train,
    cv=folds, scoring="f1",
)
# scores는 요청별 예측이 아니라 fold별 F1 3개입니다.
# std()는 이 점수들의 표준편차이며 신뢰구간을 뜻하지 않습니다.

print("fold별 F1:", scores.round(3).tolist())
print("평균 F1:", round(scores.mean(), 3))
print("표준편차:", round(scores.std(), 3))
