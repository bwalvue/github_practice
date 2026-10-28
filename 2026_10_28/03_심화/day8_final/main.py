"""실습 07: 확정 계획의 새 Pipeline으로 개발80 학습·최종 평가"""
from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np
import sklearn
import xgboost
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from preprocessing import CANDIDATES, FEATURES, NUM
from evaluate import summarize

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
# plan.py의 확정 계획 재사용, 최종 평가 단계의 후보·임계값 탐색 없음
plan = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
# 계획에 고정된 train+valid 순서로 개발80 구성, 입력과 정답 분리
develop = raw.loc[plan["train_ids"] + plan["valid_ids"]]
X_develop = develop[plan["features"]]
# 제공: 확정 파라미터의 새 Pipeline 생성, CV에서 학습한 모델의 재사용 제외
model = clone(CANDIDATES[plan["selected"]])
# 현재 후보의 기본값보다 저장된 선택 시점의 모델 파라미터 우선 적용
model["model"].set_params(**plan["model_params"])
# TODO H07a: 확정 모델을 개발80의 입력과 정답으로 학습하세요.
# 선택된 설정의 새 Pipeline을 개발 80건으로 재학습
# 저장·복원 후에도 같은 공식 적용을 위한 파생 변환의 Pipeline 포함
model.fit(X_develop, develop["failed"])
# TODO 끝
# 설정 저장·개발 학습 완료 후 test 입력·정답 선택
test = raw.loc[plan["test_ids"]]
X_test = test[plan["features"]]
positive_column = model.classes_.tolist().index(1)
# TODO H07b: 실패1의 확률을 구하세요. Feature Engineering도 저장될 Pipeline 안에서 적용됩니다.
# classes_에서 찾은 실패 1의 확률 열 선택
# test로 재학습하지 않고 개발 자료의 전처리 기준 적용
probability = model.predict_proba(X_test)[:, positive_column]
# TODO 끝
# 계획에 기록한 임계값의 엄격한 초과 규칙 적용, 같은 값은 성공 0으로 처리
prediction = (probability > plan["threshold"]).astype(int)
metrics = summarize(test["failed"], prediction, probability)
# 동일 요청 ID에 실제값·확률·판정 연결로 오류 사례 추적 지원
result = pd.DataFrame({"request_id": X_test.index, "actual": test["failed"].to_numpy(),
                       "probability_failed": probability, "predicted": prediction})
result.to_csv(OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
# 실제값과 판정이 다른 행만 분리하여 FN·FP 검토용 자료 생성
result.loc[result["actual"] != result["predicted"]].to_csv(OUT / "error_cases.csv", index=False, encoding="utf-8-sig")
(OUT / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
# 직접 학습한 Pipeline만 저장·복원, 출처가 불명확한 외부 파일 제외
joblib.dump(model, OUT / "pipeline.joblib")
# 저장 모델의 버전 의존성 확인을 위한 실행 환경 기록
environment = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
               "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}
(OUT / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
# 제공: 군집은 train60 수치2열만 사용, 최종 분류의 선택 기준과 별도 계산
X_cluster = raw.loc[plan["train_ids"], NUM]
cluster = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                    ("cluster", KMeans(n_clusters=3, n_init=10, random_state=42))])
cluster.fit(X_cluster)
labels = cluster["cluster"].labels_
# 표준화된 중심을 원래 단위로 복원하여 글자 수·도구 수 해석 지원
centers = cluster["scaler"].inverse_transform(cluster["cluster"].cluster_centers_)
profile = pd.DataFrame(centers, columns=NUM)
profile.insert(0, "cluster", range(3))
profile["n"] = np.bincount(labels, minlength=3)
profile.to_csv(OUT / "cluster_profile.csv", index=False, encoding="utf-8-sig")
# silhouette는 원단위 중심이 아닌 학습 당시의 표준화 공간에서 계산
cluster_score = silhouette_score(cluster[:-1].transform(X_cluster), labels)
print("확정 모델:", plan["selected"], "/ 개발80·최종20")
print(pd.DataFrame([metrics]).round(4).to_string(index=False))
print("K=3 군집")
print(profile.round(3).to_string(index=False))
print("군집 silhouette:", round(cluster_score, 3))
print("저장: outputs/test_predictions.csv, test_metrics.json, pipeline.joblib 등")
print("최종 성능을 본 뒤 다시 탐색하지 않습니다. 다음은 restore.py로 재현 확인입니다.")
