"""실습 01 응용: 모델 예시와 전부 성공으로 예측하는 기준선을 비교합니다."""
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
y = [0]*14 + [1]*6
predictions = {"example": [0]*10 + [1]*4 + [0]*4 + [1]*2, "all_success": [0]*20}
rows = []
for name, pred in predictions.items():
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    # TODO A01: 현재 후보의 F1을 계산하세요. 분모 0은 0으로 표시합니다.
    # 정답 예시 A01 (아래 임시 대입문을 다음 한 줄로 교체):
    # f1 = f1_score(y, pred, zero_division=0)
    # 해설: 반복문에서 pred가 현재 후보의 예측으로 바뀌므로 각 후보에 같은 계산을 적용합니다.
    # 실패1을 양성으로 두면 example은 TP=2·FP=4·FN=4로 F1=4/12, 약 0.333입니다.
    # all_success는 실패를 한 건도 발견하지 않아 TP=0·FN=6이며 F1=0입니다.
    # 전부 성공이라고 해도 Accuracy=14/20=0.70이므로 정확도와 실제 실패 발견 수를 함께 읽습니다.
    # zero_division=0은 F1 계산의 분모가 0일 때 0으로 표시하는 정책입니다.
    f1 = 0.0
    # TODO 끝
    rows.append({"model": name, "Accuracy": accuracy_score(y, pred),
                 "Precision": precision_score(y, pred, zero_division=0),
                 "Recall": recall_score(y, pred, zero_division=0), "F1": f1,
                 "FP": fp, "FN": fn, "TP": tp})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("정확도가 높은 all_success도 실패 6건을 하나도 찾지 못합니다.")
