"""실습 05: train 12건의 글자 수 한 열로 K=2 군집화"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "requests.csv")
# 제공: 이전 수업과 같은 seed·층화·행 순서 유지, test 4건은 최종 평가까지 보관
develop, test = train_test_split(raw, test_size=0.2, stratify=raw["failed"], random_state=42)
train, valid = train_test_split(develop, test_size=0.25, stratify=develop["failed"], random_state=42)

from sklearn.cluster import KMeans
# 같은 단위의 수치 한 열만 사용하므로 표준화 생략, 정답·실행시간 제외
model = KMeans(n_clusters=2, n_init=10, random_state=42)
# TODO B05: KMeans를 train의 input_chars 한 열로 학습하세요. 정답 y는 전달하지 않습니다.
# train 12건의 수치 한 열만 사용, 정답·valid·test 제외
# 2차원 입력 유지를 위한 이중 대괄호 사용
model.fit(train[["input_chars"]])
# TODO 끝
result = train[["request_id", "input_chars"]].copy()
result["cluster"] = model.labels_
print("군집별 개수:", result["cluster"].value_counts().sort_index().to_dict())
print("중심(글자):", model.cluster_centers_.round(1).ravel().tolist())
result.to_csv(HERE / "clusters.csv", index=False, encoding="utf-8-sig")
print("저장: clusters.csv (군집 번호는 성공·실패 라벨이 아닙니다.)")
