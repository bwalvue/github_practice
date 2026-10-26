"""같은 학습·검증 분할에서 KNN의 이웃 수만 바꾸어 비교합니다."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "requests.csv")

# K에 따른 차이만 비교하려면 모든 후보가 같은 행을 학습·평가해야 하므로 분할을 반복문 밖에 둡니다.
# random_state를 고정해 재실행 시에도 같은 분할을 사용하고, stratify로 라벨 비율을 고려합니다.
develop, test = train_test_split(
    raw, test_size=0.2, random_state=42, stratify=raw["failed"]
)
train, valid = train_test_split(
    develop, test_size=0.25, random_state=42, stratify=develop["failed"]
)

# train["input_chars"]는 1차원 Series입니다. 이중 대괄호로 선택해야 모델이 받는 2차원 표를 유지합니다.
X_train = train[["input_chars"]]
X_valid = valid[["input_chars"]]

for k in [1, 3, 5]:

    # TODO B06: 고정된 이웃 수 1을 반복 변수 k로 바꾸세요.
    # 풀이: 출력할 k만 바꾸면 같은 K=1 모델을 반복할 수 있으므로, 생성 인자에도 k를 전달합니다.
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    model.fit(X_train, train["failed"])
    prediction = model.predict(X_valid)

    # 반복 변수 대신 실제 모델의 n_neighbors를 출력해, 비교한 설정과 기록된 K가 일치하는지 확인합니다.
    print(
        "K:", model[-1].n_neighbors,
        "valid F1:", round(
            f1_score(valid["failed"], prediction, zero_division=0), 3
        ),
    )

print("분할:", len(train), len(valid), len(test))
