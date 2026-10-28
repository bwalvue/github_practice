"""실습 02 응용: 가상 확률의 임계값 비교, 최종 모델의 임계값 탐색과 무관"""
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
    # 임계값 초과만 실패 1로 변환, 같은 값은 성공 0으로 처리
    # 가상 확률의 원리 비교이며 최종 모델 임계값 재선택과 무관
    prediction = (score > threshold).astype(int)
    # TODO 끝
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    rows.append({"threshold": threshold, "FP": fp, "FN": fn, "F1": f1_score(y, prediction),
                 "assumed_cost": fp + 5*fn, "AUC": roc_auc_score(y, score)})
result = pd.DataFrame(rows)
print(result.round(3).to_string(index=False))
result.to_csv(OUT / "threshold_comparison.csv", index=False, encoding="utf-8-sig")
print("cost는 FP 비용1, FN 비용5라는 가정입니다. 최종 평가 임계값은 사전에 0.5로 고정합니다.")
