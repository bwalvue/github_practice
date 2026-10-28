"""실습 02 응용: 가상 확률로 임계값만 바꿉니다. 최종 모델의 임계값을 탐색하는 코드가 아닙니다."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
y = np.array([0, 0, 1, 1])
score = np.array([0.10, 0.40, 0.35, 0.80])
rows = []
for threshold in [0.5, 0.3]:
    # TODO A02: 확률이 threshold보다 클 때만 실패1로 바꾸세요. 같은 값은 성공0입니다.
    # 정답 예시 A02 (아래 임시 대입문을 다음 한 줄로 교체):
    # prediction = (score > threshold).astype(int)
    # 해설: 원소별 비교는 True·False 배열을 만들고, astype(int)는 이를 1·0으로 바꿉니다.
    # 규칙은 >이므로 점수가 임계값과 정확히 같으면 성공0입니다. >=로 바꾸지 않습니다.
    # 0.5에서는 [0, 0, 0, 1]로 FP=0·FN=1, 0.3에서는 [0, 1, 1, 1]로 FP=1·FN=0입니다.
    # 가정한 비용 FP+5*FN은 각각 5와 1입니다. 같은 score를 쓰므로 AUC는 둘 다 0.75입니다.
    # 이 원리 예제의 비교로 최종 프로젝트의 임계값 0.5를 변경하지 않습니다.
    prediction = np.zeros(len(y), dtype=int)
    # TODO 끝
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    rows.append({"threshold": threshold, "FP": fp, "FN": fn, "F1": f1_score(y, prediction),
                 "assumed_cost": fp + 5*fn, "AUC": roc_auc_score(y, score)})
result = pd.DataFrame(rows)
print(result.round(3).to_string(index=False))
result.to_csv(OUT / "threshold_comparison.csv", index=False, encoding="utf-8-sig")
print("cost는 FP 비용1, FN 비용5라는 가정입니다. 최종 평가 임계값은 사전에 0.5로 고정합니다.")
