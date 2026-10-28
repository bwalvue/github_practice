"""실습 07~08: 06번의 설정을 그대로 읽어 개발 16건으로 학습하고 test 4건을 최종 평가합니다."""
from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
HERE = Path(__file__).resolve().parent
# 제공: 06번을 먼저 완료하세요. 이 프로그램은 CV·설정 선택을 다시 하지 않습니다.
plan = json.loads((HERE / "selection.json").read_text(encoding="utf-8"))
raw = pd.read_csv(HERE / "requests.csv").set_index("request_id")
develop = raw.loc[plan["train_ids"] + plan["valid_ids"]].copy()
test = raw.loc[plan["test_ids"]].copy()
# 제공: 행별 공식은 새 요청에도 동일하게 적용합니다. test로 통계나 설정을 학습하지 않습니다.
for table in (develop, test):
    table["log_input_chars"] = np.log1p(table["input_chars"])
model = DecisionTreeClassifier(max_depth=plan["max_depth"], random_state=plan["random_state"])
# TODO B07a: 개발 자료 16건의 선택된 입력과 failed로 학습하세요. test는 넣지 않습니다.
# 정답 예시 B07a (아래 pass를 다음 한 줄로 교체):
# model.fit(develop[plan["features"]], develop["failed"])
# 해설: develop은 계획에 기록된 train 12건과 valid 4건을 합친 개발 자료입니다.
# plan["features"]는 06번에서 선택한 입력 열 목록입니다. 여기서 후보나 특성을 다시 고르지 않습니다.
# 입력과 같은 행 순서의 failed를 정답으로 전달해 새 트리를 학습합니다. test 4건은 제외합니다.
# 학습 후에야 model.classes_가 생기며, 아래에서 실패 클래스 1이 있는 확률 열을 찾을 수 있습니다.
pass
# TODO 끝
# 0·1 두 클래스가 있는 정상 자료입니다. classes_에서 양성 1의 위치를 확인합니다.
positive_column = model.classes_.tolist().index(1)
# TODO B07b: test의 실패 확률을 구하세요. predict_proba의 positive_column 열만 선택합니다.
# 정답 예시 B07b (아래 임시 대입문을 다음 한 줄로 교체):
# probability = model.predict_proba(test[plan["features"]])[:, positive_column]
# 해설: predict_proba는 학습된 트리에 test의 입력만 넣어 각 클래스의 확률을 구합니다.
# 반환값은 4행·2열이며 열 순서는 model.classes_를 따릅니다.
# [:, positive_column]은 모든 행에서 실패 클래스 1의 열 하나를 골라 1차원 배열로 만듭니다.
# test의 정답은 예측 입력에 넣지 않습니다. 아래에서 확률이 계획의 임계값을 초과할 때만 실패1로 판정합니다.
probability = np.zeros(len(test))
# TODO 끝
prediction = (probability > plan["threshold"]).astype(int)
# TODO B07c: F1에는 0/1 예측값, AUC에는 실패 확률을 넣으세요.
# 정답 예시 B07c (아래 임시 대입문을 다음 두 줄로 교체):
# f1 = f1_score(test["failed"], prediction, zero_division=0)
# auc = roc_auc_score(test["failed"], probability)
# 해설: 두 함수의 첫 인수는 test의 실제 정답입니다. 평가할 때만 정답과 예측을 비교합니다.
# F1은 임계값으로 결정한 0/1 판정의 Precision·Recall 균형을, AUC는 연속 확률의 순위를 평가합니다.
# 따라서 F1에 probability를 직접 넣거나 AUC에 prediction을 넣지 않습니다.
# zero_division=0은 F1의 분모가 0인 경우의 정책입니다. AUC에는 이 옵션을 전달하지 않습니다.
# 최종 결과가 나온 뒤 같은 test 점수로 06번 후보·특성·임계값을 다시 고르지 않습니다.
f1 = auc = 0.0
# TODO 끝
metrics = {"n": len(test), "Accuracy": accuracy_score(test["failed"], prediction),
           "Precision": precision_score(test["failed"], prediction, zero_division=0),
           "Recall": recall_score(test["failed"], prediction, zero_division=0),
           "F1": f1, "ROC_AUC": auc, "TP": int(((test["failed"] == 1) & (prediction == 1)).sum()),
           "actual_failed": int(test["failed"].sum()), "threshold": plan["threshold"]}
result = pd.DataFrame({"request_id": test.index, "actual": test["failed"].to_numpy(),
                       "probability_failed": probability, "predicted": prediction})
result.to_csv(HERE / "test_predictions.csv", index=False, encoding="utf-8-sig")
(HERE / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print("개발/최종평가:", len(develop), len(test))
print(result.round(3).to_string(index=False))
print("혼동행렬:", confusion_matrix(test["failed"], prediction, labels=[0, 1]).tolist())
print(f"Accuracy={metrics['Accuracy']:.3f}, Precision={metrics['Precision']:.3f}, Recall={metrics['Recall']:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
print("저장: test_predictions.csv, test_metrics.json")
print("08번은 이 설정 그대로 재실행합니다. test를 보고 06번 설정을 다시 고르지 않습니다.")
