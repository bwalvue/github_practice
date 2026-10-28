"""실습 06: train 내부 3-fold로 특성 비교·계획 기록, test 평가 제외"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "requests.csv")
# 제공: 이전 수업과 같은 seed·층화·행 순서 유지, test 4건은 최종 평가까지 보관
develop, test = train_test_split(raw, test_size=0.2, stratify=raw["failed"], random_state=42)
train, valid = train_test_split(develop, test_size=0.25, stratify=develop["failed"], random_state=42)

import numpy as np
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
# 두 파생 연산 모두 자기 행의 글자 수만 사용, 다른 행의 통계·정답 제외
train = train.copy()
train["log_input_chars"] = np.log1p(train["input_chars"])
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
choices = {"original": ["input_chars"], "with_log": ["input_chars", "log_input_chars"]}
rows = []
for name, columns in choices.items():
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    # TODO B06: 같은 folds로 model을 교차검증하세요. train만 넣고 scoring은 f1입니다.
    # 동일 train 12건의 3-fold F1로 원본·로그 특성 후보 비교
    # 자기 행만 사용하는 로그 변환이므로 CV 이전 계산 가능
    scores = cross_val_score(
        model, train[columns], train["failed"],
        cv=folds, scoring="f1", n_jobs=1,
    )
    # TODO 끝
    rows.append({"features": name, "CV_F1": scores.mean(), "CV_std": scores.std()})
comparison = pd.DataFrame(rows)
# 제공: CV 평균 F1 최대로 선택, 정확한 동점은 original 우선
selected = comparison.loc[comparison["CV_F1"].idxmax(), "features"]
plan = {"model": "DecisionTreeClassifier", "max_depth": 3, "random_state": 42,
        "features": choices[selected], "threshold": 0.5, "rule": "score > threshold",
        "positive_label": 1, "selection": "train12 3-fold mean F1; tie: original first",
        "train_ids": train["request_id"].tolist(), "valid_ids": valid["request_id"].tolist(),
        "test_ids": test["request_id"].tolist()}
comparison.to_csv(HERE / "feature_comparison.csv", index=False, encoding="utf-8-sig")
(HERE / "selection.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
print(comparison.round(3).to_string(index=False))
print("선택:", selected, "/ 임계값 0.5 초과일 때 실패")
print("설정 기록: selection.json / test는 아직 예측하지 않았습니다.")
