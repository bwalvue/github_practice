"""실습 07~08 응용: 저장된 계획으로 개발80 재학습·test20 최종 평가"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
# 06번의 선택 결과 재사용으로 후보·임계값 탐색과 최종 평가 단계 분리
plan = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
ids = plan["split_ids"]
develop = raw.loc[ids["train"] + ids["valid"]]
test = raw.loc[ids["test"]]
# 계획의 입력 열만 선택하여 정답·실행시간의 입력 혼입 방지
X_develop, X_test = develop[plan["features"]].copy(), test[plan["features"]].copy()
# 계획의 앞 두 열은 수치, 뒤 두 열은 범주로 전처리 경로 분리
NUM, CAT = plan["features"][:2], plan["features"][2:]
# 제공: 새 입력에도 같은 행별 공식 적용, 학습 통계는 fit(develop)에서만 계산
# Logistic_FE 선택 시에만 파생 열 추가, 원본 후보는 기존 입력 유지
if plan["derived"]:
    for part in (X_develop, X_test):
        part["log_input_chars"] = np.log1p(part["input_chars"])
        part["chars_per_slot"] = part["input_chars"] / (1.0 + part["allowed_tools"])
    # 파생 열의 전처리 누락 방지를 위한 수치 열 목록 확장
    NUM = NUM + ["log_input_chars", "chars_per_slot"]
# 수치 열은 중앙값·표준화, 범주 열은 최빈값·원핫 인코딩으로 처리
pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler())]), NUM),
    # 새 요청의 미등록 범주도 오류 없이 처리하기 위한 ignore 설정
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
])
# 전처리와 분류기를 묶어 학습·추론에 같은 처리 순서 적용
model = Pipeline([("pre", pre), ("model", LogisticRegression(C=plan["C"], max_iter=plan["max_iter"]))])
# TODO A07a: 새 Pipeline을 X_develop·failed로 학습하세요. 기존 train 통계를 재사용하지 않고 개발80에서 새로 fit합니다.
# 새 Pipeline 전체를 개발 80건으로 fit하여 전처리 통계·모델 재학습
# 선택 계획 유지 및 test 20건의 학습 제외
model.fit(X_develop, develop["failed"])
# TODO 끝
positive_column = model.classes_.tolist().index(1)
# TODO A07b: 확정 모델로 test의 실패 확률을 구하세요. test는 fit하지 않습니다.
# 개발 자료에서 배운 전처리 적용 후 실패 1의 확률 선택
# 계획의 0.5 초과 판정을 위한 연속 확률 반환
probability = model.predict_proba(X_test)[:, positive_column]
# TODO 끝
prediction = (probability > plan["threshold"]).astype(int)
# 실제 행·예측 열 및 클래스 [0, 1] 고정으로 TN·FP·FN·TP 순서 확보
tn, fp, fn, tp = confusion_matrix(test["failed"], prediction, labels=[0, 1]).ravel()
metrics = {"n": len(test), "Accuracy": accuracy_score(test["failed"], prediction),
           "Precision": precision_score(test["failed"], prediction, zero_division=0),
           "Recall": recall_score(test["failed"], prediction, zero_division=0),
           "F1": f1_score(test["failed"], prediction, zero_division=0),
           "ROC_AUC": roc_auc_score(test["failed"], probability),
           "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "actual_failed": int(test["failed"].sum())}
# 요청별 오판 추적을 위한 실제값·확률·판정의 동일 행 순서 보존
result = pd.DataFrame({"request_id": test.index, "actual": test["failed"].to_numpy(),
                       "probability_failed": probability, "predicted": prediction})
result.to_csv(OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
(OUT / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print("확정 모델:", plan["selected"], "/ 학습80·최종20")
print(pd.DataFrame([metrics]).round(4).to_string(index=False))
print("오류 사례")
print(result.loc[result["actual"] != result["predicted"]].head(3).round(3).to_string(index=False))
print("저장: outputs/test_predictions.csv, test_metrics.json")
print("08번에서는 이 설정·원본을 그대로 재실행합니다. test를 보고 후보를 다시 고르지 않습니다.")
