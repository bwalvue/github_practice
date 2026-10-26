"""실습08: main.py로 본인이 생성한 모델만 복원합니다. 재학습하지 않습니다."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from preprocessing import FEATURES

HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
split = pd.read_csv(HERE / "data/split_manifest.csv")
ids = split.loc[split["split"] == "valid"].sort_values("position")["request_id"]
X_valid = raw.loc[ids, FEATURES]
selection = json.loads((HERE / "outputs/selection.json").read_text(encoding="utf-8"))
saved = pd.read_csv(HERE / "outputs/validation_predictions.csv")

# TODO H08: main.py로 직접 만든 models.joblib을 읽으세요.
# 풀이: main.py에서 저장한 Pipeline 딕셔너리를 읽습니다. 다시 fit하면 저장 상태의 복원이 아니라 재학습을 비교하게 됩니다.
models = joblib.load(HERE / "outputs/models.joblib")
for task in ["regression", "classification"]:
    prediction = models[task].predict(X_valid)

    # 예측 CSV에는 여러 후보가 섞여 있으므로 선택 모델만 고른 뒤 ID 순서를 맞춰 같은 요청끼리 비교합니다.
    reference = saved[(saved["task"] == task) & (saved["model"] == selection[task])].set_index("request_id").loc[ids]

    # CSV를 거친 실수는 미세한 표현 차이가 있을 수 있어, 원소의 완전 동일성 대신 allclose로 수치가 가까운지 비교합니다.
    print(task, "복원 예측 일치:", bool(np.allclose(prediction, reference["predicted"])))
print("fit을 호출하지 않았고 test도 사용하지 않았습니다.")
