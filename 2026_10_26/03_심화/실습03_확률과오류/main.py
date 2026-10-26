"""실습 03 심화: 공통 데이터의 확률과 별도 가상 확률 예제를 구분합니다."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
split = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = split.loc[split["split"] == "train"].sort_values("position")["request_id"]
valid_ids = split.loc[split["split"] == "valid"].sort_values("position")["request_id"]
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]
features = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train = train[features]
X_valid = valid[features]

import numpy as np
from preprocessing import build_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

model = build_model(LogisticRegression(C=1.0, max_iter=1000))
model.fit(X_train, train["failed"])
actual_prediction = model.predict(X_valid)
print("공통 valid 혼동행렬")
print(confusion_matrix(valid["failed"], actual_prediction, labels=[0, 1]))

# 아래 여섯 확률은 위 Logistic 모델의 출력과 별개입니다. 임계값 변경 효과를 고정된 작은 예제로 계산합니다.
truth = np.array([0, 0, 0, 1, 1, 1])
probability = np.array([0.10, 0.40, 0.70, 0.35, 0.60, 0.90])
rows = []
for threshold in [0.5, 0.3]:

    # TODO H03: 고정값 0.5 대신 반복 변수 threshold를 사용하세요.
    # 풀이: 표의 threshold 이름만 바꾸지 않고 실제 비교식에도 전달합니다. astype(int)는 판정 결과를 0/1 라벨로 만듭니다.
    # 0.5→0.3이면 확률 0.40의 성공도 실패로 잡지만, 확률 0.35의 실제 실패를 새로 찾아 FP와 FN이 함께 바뀝니다.
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    rows.append({"threshold": threshold, "FP": fp, "FN": fn, "TP": tp,
                 "F1": f1_score(truth, prediction, zero_division=0)})
comparison = pd.DataFrame(rows)
out = HERE / "outputs"
out.mkdir(exist_ok=True)

comparison.to_csv(out / "threshold_comparison.csv", index=False, encoding="utf-8-sig")
print("가상 확률 6개의 임계값 비교")
print(comparison.round(3).to_string(index=False))
