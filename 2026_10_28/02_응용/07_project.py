"""실습 07~08 응용: 06번 계획을 읽어 개발80으로 새 Pipeline을 학습하고 test20을 평가합니다."""
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
plan = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
ids = plan["split_ids"]
develop = raw.loc[ids["train"] + ids["valid"]]
test = raw.loc[ids["test"]]
X_develop, X_test = develop[plan["features"]].copy(), test[plan["features"]].copy()
NUM, CAT = plan["features"][:2], plan["features"][2:]
# 제공: 같은 행별 공식을 새 입력에도 적용합니다. 학습 통계는 아래 fit(develop)에서만 계산됩니다.
if plan["derived"]:
    for part in (X_develop, X_test):
        part["log_input_chars"] = np.log1p(part["input_chars"])
        part["chars_per_slot"] = part["input_chars"] / (1.0 + part["allowed_tools"])
    NUM = NUM + ["log_input_chars", "chars_per_slot"]
pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler())]), NUM),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
])
model = Pipeline([("pre", pre), ("model", LogisticRegression(C=plan["C"], max_iter=plan["max_iter"]))])
# TODO A07a: 새 Pipeline을 X_develop·failed로 학습하세요. 기존 train 통계를 재사용하지 않고 개발80에서 새로 fit합니다.
# 정답 예시 A07a (아래 pass를 다음 한 줄로 교체):
# model.fit(X_develop, develop["failed"])
# 해설: develop은 계획에 기록한 train60·valid20을 합친 80건이며 입력과 정답의 행 순서가 같습니다.
# 새 Pipeline 전체를 fit하므로 결측 대체·표준화·범주 목록과 모델을 개발80건에서 새로 학습합니다.
# 06번의 선택·특성·C·임계값을 그대로 사용하고, 후보 비교나 CV를 여기서 반복하지 않습니다.
# test20의 입력·정답은 fit에 넣지 않습니다. 이후 test에는 이 학습된 전처리 기준만 적용합니다.
pass
# TODO 끝
positive_column = model.classes_.tolist().index(1)
# TODO A07b: 확정 모델로 test의 실패 확률을 구하세요. test는 fit하지 않습니다.
# 정답 예시 A07b (아래 임시 대입문을 다음 한 줄로 교체):
# probability = model.predict_proba(X_test)[:, positive_column]
# 해설: X_test에는 계획에 맞는 입력만 들어 있으며 필요하면 동일한 행별 파생 공식도 적용된 상태입니다.
# predict_proba는 개발80건에서 학습한 전처리와 모델을 사용합니다. test로 다시 fit하지 않습니다.
# 반환 표의 열 순서는 model.classes_를 따르며 [:, positive_column]으로 실패1 확률 20개를 고릅니다.
# 아래 prediction은 확률이 0.5를 초과한 경우만 실패1로 바꾸고, F1은 판정·AUC는 확률로 계산합니다.
# 08번 재실행에서도 같은 설정을 유지합니다. test 결과를 보고 후보·특성·임계값을 재선택하지 않습니다.
probability = np.zeros(len(test))
# TODO 끝
prediction = (probability > plan["threshold"]).astype(int)
tn, fp, fn, tp = confusion_matrix(test["failed"], prediction, labels=[0, 1]).ravel()
metrics = {"n": len(test), "Accuracy": accuracy_score(test["failed"], prediction),
           "Precision": precision_score(test["failed"], prediction, zero_division=0),
           "Recall": recall_score(test["failed"], prediction, zero_division=0),
           "F1": f1_score(test["failed"], prediction, zero_division=0),
           "ROC_AUC": roc_auc_score(test["failed"], probability),
           "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "actual_failed": int(test["failed"].sum())}
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
