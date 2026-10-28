"""실습 04: 자기 행의 입력만 사용하는 파생 특성 생성"""
from pathlib import Path
import pandas as pd
import numpy as np
raw = pd.read_csv(Path(__file__).resolve().parent / "requests.csv")
# 원본 보존을 위한 글자 수 열 복사, 정답·실행시간 제외
features = raw[["input_chars"]].copy()
# TODO B04: np.log1p로 글자 수의 로그 변환 열을 추가하세요. 각 행의 값만 사용합니다.
# 자기 행의 글자 수에 log(1+x) 적용, 큰 값 사이의 간격 축소 목적
# 다른 행의 통계나 정답을 사용하지 않는 행별 변환
features["log_input_chars"] = np.log1p(features["input_chars"])
# TODO 끝
print(features.head().round(3).to_string(index=False))
print("행/열:", features.shape)
print("원본 열:", raw.columns.tolist())
