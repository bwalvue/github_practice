"""실습 02: 깊이만 바꾸고 같은 train·valid에서 비교합니다."""

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
from sklearn.metrics import f1_score

for depth in [1, 3]:
    # TODO B02: 반복 중인 depth를 max_depth에 연결하세요.
    # 풀이: 출력에 표시하는 depth를 실제 모델 설정에도 연결합니다.
    # 매번 새 트리를 만들고 데이터·분할·난수는 고정해 깊이만 비교합니다.
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)

    model.fit(X_train, y_train)
    train_f1 = f1_score(y_train, model.predict(X_train), zero_division=0)
    valid_f1 = f1_score(y_valid, model.predict(X_valid), zero_division=0)
    print(f"깊이 {depth}: train F1={train_f1:.3f}, valid F1={valid_f1:.3f}")
