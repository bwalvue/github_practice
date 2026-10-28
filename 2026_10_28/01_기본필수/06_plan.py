"""실습 06: test를 보지 않고 train 내부 3-fold로 두 표현을 비교하여 계획을 기록합니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "requests.csv")
# 제공: 앞 수업과 같은 seed·층화·행 순서입니다. test 4건은 마지막 평가 전까지 보관합니다.
develop, test = train_test_split(raw, test_size=0.2, stratify=raw["failed"], random_state=42)
train, valid = train_test_split(develop, test_size=0.25, stratify=develop["failed"], random_state=42)

import numpy as np
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
# 두 파생 연산 모두 자기 행의 글자 수만 사용합니다. 다른 행의 평균·정답을 보지 않습니다.
train = train.copy()
train["log_input_chars"] = np.log1p(train["input_chars"])
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
choices = {"original": ["input_chars"], "with_log": ["input_chars", "log_input_chars"]}
rows = []
for name, columns in choices.items():
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    # TODO B06: 같은 folds로 model을 교차검증하세요. train만 넣고 scoring은 f1입니다.
    # 정답 예시 B06 (아래 임시 대입문을 다음 호출로 교체):
    # scores = cross_val_score(
    #     model, train[columns], train["failed"],
    #     cv=folds, scoring="f1", n_jobs=1,
    # )
    # 해설: columns는 현재 후보의 입력 열 목록이며, failed는 그 행들의 실제 분류 정답입니다.
    # train 12건 안에서 각 fold의 8건으로 학습하고 나머지 4건으로 F1을 계산해 점수 3개를 받습니다.
    # 같은 folds와 random_state를 사용하므로 original·with_log를 같은 행 분할로 비교합니다.
    # 아래 mean은 세 점수의 평균, std는 표준편차입니다. 가장 큰 평균으로 계획을 선택합니다.
    # valid·test를 교차검증에 넣지 않으며, 이 단계에서는 test 예측이나 최종 평가를 하지 않습니다.
    scores = np.zeros(3)
    # TODO 끝
    rows.append({"features": name, "CV_F1": scores.mean(), "CV_std": scores.std()})
comparison = pd.DataFrame(rows)
# 제공: CV 평균 F1 최대를 선택합니다. 정확히 같으면 먼저 적은 original을 고릅니다.
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
