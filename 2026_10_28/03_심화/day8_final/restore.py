"""실습 08: 직접 생성한 모델의 새 프로세스 복원, 학습·후보 선택 없음"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
# 저장 모델이 참조하는 features.py의 이름 있는 함수 보존 필요
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
plan = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
expected = pd.read_csv(OUT / "test_predictions.csv")
# main.py에서 직접 생성한 모델만 복원, 신뢰할 수 없는 외부 파일 제외
model = joblib.load(OUT / "pipeline.joblib")
X = raw.loc[expected["request_id"], plan["features"]]
positive_column = model.classes_.tolist().index(1)
# TODO H08: 복원 모델로 같은 ID 순서의 실패 확률을 구하세요. fit은 호출하지 않습니다.
# 같은 ID 순서의 입력으로 저장된 전처리·모델 재사용
# 복원 검증 중 fit·후보 재선택 호출 없음
probability = model.predict_proba(X)[:, positive_column]
# TODO 끝
prediction = (probability > plan["threshold"]).astype(int)
print("ID 순서 동일:", X.index.tolist() == expected["request_id"].tolist())
print("확률 동일:", np.allclose(probability, expected["probability_failed"], rtol=1e-10, atol=1e-12))
print("예측 동일:", np.array_equal(prediction, expected["predicted"]))
print("복원은 고정 버전의 재현입니다. 점수를 보고 임계값을 다시 고르지 않습니다.")
