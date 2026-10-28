"""실습 04: 다른 행의 통계나 정답을 사용하지 않는 파생 열 하나를 만듭니다."""
from pathlib import Path
import pandas as pd
import numpy as np
raw = pd.read_csv(Path(__file__).resolve().parent / "requests.csv")
# 원본 파일은 덮어쓰지 않습니다. 글자 수만 복사하고 정답·실행시간은 선택하지 않습니다.
features = raw[["input_chars"]].copy()
# TODO B04: np.log1p로 글자 수의 로그 변환 열을 추가하세요. 각 행의 값만 사용합니다.
# 정답 예시 B04 (아래 임시 대입문을 다음 한 줄로 교체):
# features["log_input_chars"] = np.log1p(features["input_chars"])
# 해설: np.log1p(x)는 자연로그 log(1+x)를 계산합니다. 글자 수 0도 log(1)=0이 됩니다.
# 큰 값의 간격을 줄이는 표현이며, 예를 들어 100은 약 4.615로 바뀝니다.
# 오른쪽은 모든 행의 input_chars에 같은 공식을 적용하고, 왼쪽은 새 열에 결과를 저장합니다.
# features는 복사본이므로 raw와 원본 CSV는 바뀌지 않으며 결과 모양은 20행·2열입니다.
# 각 행의 입력만 사용하므로 다른 행의 통계를 학습하지 않습니다. failed·elapsed_ms도 쓰지 않습니다.
features["log_input_chars"] = 0.0
# TODO 끝
print(features.head().round(3).to_string(index=False))
print("행/열:", features.shape)
print("원본 열:", raw.columns.tolist())
