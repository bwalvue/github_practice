"""실습 06 응용: 04번에서 만든 CV 표만 읽어 선택하고 평가 계획을 기록합니다."""
from pathlib import Path
import pandas as pd
import json
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
# 먼저 04_features.py를 완료하세요. valid_F1을 이용해 후보를 재선택하지 않습니다.
table = pd.read_csv(OUT / "feature_comparison.csv")
# TODO A06: CV_F1이 가장 큰 행의 model 이름을 선택하세요. 정확한 동점은 표의 첫 후보를 사용합니다.
# 정답 예시 A06 (아래 임시 대입문을 다음 한 줄로 교체):
# selected = table.loc[table["CV_F1"].idxmax(), "model"]
# 해설: idxmax()는 CV_F1이 가장 큰 행의 인덱스를 반환하고, loc은 그 행의 model 이름을 읽습니다.
# 정확히 같은 최대값이 여러 개면 첫 행을 고르므로 원래 후보 순서의 동점 규칙을 따릅니다.
# 선택에는 CV_F1만 사용합니다. valid_F1은 진단용이며 test는 아직 예측하지 않습니다.
# 제공 기준에서 Logistic_FE가 선택되지만 이름을 정답으로 고정하지 않고 실제 비교표에서 찾습니다.
# 아래 코드는 선택된 이름에 따라 파생 열 사용 여부와 학습·평가 계획을 selection.json에 기록합니다.
selected = "Logistic"
# TODO 끝
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
ids = {name: manifest.loc[manifest["split"] == name].sort_values("position")["request_id"].tolist()
       for name in ["train", "valid", "test"]}
plan = {"selected": selected, "C": 1, "max_iter": 1000,
        "features": ["input_chars", "allowed_tools", "task_type", "priority"],
        "derived": selected == "Logistic_FE", "threshold": 0.5,
        "rule": "score > threshold", "positive_label": 1,
        "selection": "train60 5-fold mean F1; exact tie: Logistic first", "split_ids": ids}
(OUT / "selection.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
print("선택:", selected)
print("설정: C=1 / 임계값 0.5 초과 / 개발80으로 재학습 / test20 최종 확인")
print("저장: outputs/selection.json (최종평가 이후 변경하지 않습니다.)")
