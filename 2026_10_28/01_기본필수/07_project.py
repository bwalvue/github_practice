"""실습 07~08: 저장된 설정으로 개발 16건 학습·test 4건 최종 평가"""
from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# 터미널 작업 위치와 무관한 입력·결과 경로 확보를 위한 파일 위치 기준
HERE = Path(__file__).resolve().parent
# 제공: 06번의 선택 계획 재사용, CV·설정 선택 반복 없음
plan = json.loads((HERE / "selection.json").read_text(encoding="utf-8"))
raw = pd.read_csv(HERE / "requests.csv").set_index("request_id")
# 계획의 ID·순서대로 개발16·test4 구성, 분할 재추출 방지
develop = raw.loc[plan["train_ids"] + plan["valid_ids"]].copy()
test = raw.loc[plan["test_ids"]].copy()
# 제공: 새 요청에도 같은 행별 공식 적용, test를 이용한 통계·설정 학습 제외
for table in (develop, test):
    table["log_input_chars"] = np.log1p(table["input_chars"])
model = DecisionTreeClassifier(max_depth=plan["max_depth"], random_state=plan["random_state"])
# TODO B07a: 개발 자료 16건의 선택된 입력과 failed로 학습하세요. test는 넣지 않습니다.
# 선택 계획의 입력 열로 train 12건과 valid 4건을 합쳐 학습
# 후보·임계값 유지 및 test 4건의 학습 제외
model.fit(develop[plan["features"]], develop["failed"])
# TODO 끝
# 정상 자료의 두 클래스 중 실패 1의 확률 열 위치 확인
positive_column = model.classes_.tolist().index(1)
# TODO B07b: test의 실패 확률을 구하세요. predict_proba의 positive_column 열만 선택합니다.
# classes_의 열 순서에 맞춰 실패 1의 확률 선택
# test 정답의 입력 혼입 방지를 위해 선택된 특성만 전달
probability = model.predict_proba(test[plan["features"]])[:, positive_column]
# TODO 끝
# 임계값 초과만 실패 1로 분기, 같은 값은 성공 0으로 처리
prediction = (probability > plan["threshold"]).astype(int)
# TODO B07c: F1에는 0/1 예측값, AUC에는 실패 확률을 넣으세요.
# F1에는 0/1 판정, AUC에는 순위 정보가 있는 연속 확률 사용
# F1 분모 0인 경우 0 반환을 위한 zero_division=0 지정
f1 = f1_score(test["failed"], prediction, zero_division=0)
auc = roc_auc_score(test["failed"], probability)
# TODO 끝
# 전체 정답률과 실패 탐지 성능 구분을 위한 지표 묶음
metrics = {"n": len(test), "Accuracy": accuracy_score(test["failed"], prediction),
           "Precision": precision_score(test["failed"], prediction, zero_division=0),
           "Recall": recall_score(test["failed"], prediction, zero_division=0),
           "F1": f1, "ROC_AUC": auc, "TP": int(((test["failed"] == 1) & (prediction == 1)).sum()),
           "actual_failed": int(test["failed"].sum()), "threshold": plan["threshold"]}
# 요청 ID별 실제값·확률·판정 연결로 오판 추적 지원
result = pd.DataFrame({"request_id": test.index, "actual": test["failed"].to_numpy(),
                       "probability_failed": probability, "predicted": prediction})
# 같은 이름의 예측 CSV·지표 JSON 갱신으로 재실행 결과 기록
result.to_csv(HERE / "test_predictions.csv", index=False, encoding="utf-8-sig")
(HERE / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print("개발/최종평가:", len(develop), len(test))
print(result.round(3).to_string(index=False))
print("혼동행렬:", confusion_matrix(test["failed"], prediction, labels=[0, 1]).tolist())
print(f"Accuracy={metrics['Accuracy']:.3f}, Precision={metrics['Precision']:.3f}, Recall={metrics['Recall']:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
print("저장: test_predictions.csv, test_metrics.json")
print("08번은 이 설정 그대로 재실행합니다. test를 보고 06번 설정을 다시 고르지 않습니다.")
