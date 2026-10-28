"""실습 06 응용: 04번의 CV 비교표 기반 선택·평가 계획 기록"""
from pathlib import Path
import pandas as pd
import json
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
# 04_features.py의 비교표 사용, 진단용 valid_F1을 통한 후보 재선택 제외
table = pd.read_csv(OUT / "feature_comparison.csv")
# TODO A06: CV_F1이 가장 큰 행의 model 이름을 선택하세요. 정확한 동점은 표의 첫 후보를 사용합니다.
# 진단용 valid 점수 대신 train 내부 CV 평균 F1로 선택
# 정확한 동점은 idxmax()의 첫 행 선택 규칙 적용
selected = table.loc[table["CV_F1"].idxmax(), "model"]
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
