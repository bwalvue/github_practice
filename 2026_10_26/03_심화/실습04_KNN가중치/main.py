"""실습 04 심화: 이전 응용의 거리 가중치를 표로 설명합니다."""
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

X = np.array([[1], [2], [4], [7], [9]])
y = np.array([10, 20, 40, 70, 90])
plain = KNeighborsRegressor(n_neighbors=3).fit(X, y)

# indices는 X의 행 위치입니다. X와 y에 같은 위치를 적용해야 거리와 이웃의 정답이 맞게 연결됩니다.
distance, indices = plain.kneighbors([[3.2]])
print(pd.DataFrame({"input": X[indices[0], 0], "distance": distance[0], "target": y[indices[0]]}).round(3).to_string(index=False))

# TODO H04: 가까운 이웃에 큰 가중치를 주도록 변경하세요.
# 풀이: 세 이웃은 유지하고 weights만 distance로 바꿉니다. 거리 0.8인 입력 4의 정답 40에 가장 큰 비중을 줍니다.
weighted = KNeighborsRegressor(n_neighbors=3, weights="distance")
weighted.fit(X, y)
print("uniform:", plain.predict([[3.2]]).round(2))
print("distance:", weighted.predict([[3.2]]).round(2))

# 모델과 같은 1/거리 가중치를 사용해 손계산하면, 단순 평균과 결과가 달라지는 이유를 수치로 확인할 수 있습니다.
print("역거리 가중 손계산:", round(np.average(y[indices[0]], weights=1 / distance[0]), 2))
