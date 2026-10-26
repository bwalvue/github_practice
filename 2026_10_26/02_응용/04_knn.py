"""같은 세 이웃에서 단순 평균과 거리 가중 평균을 비교합니다."""
from sklearn.neighbors import KNeighborsRegressor

X = [[1], [2], [4], [7], [9]]
y = [10, 20, 40, 70, 90]

for weight in ["uniform", "distance"]:

    # TODO A04: weights에 반복 변수 weight를 넣으세요.
    # 풀이: 같은 세 이웃을 두고 가중 방식만 비교해야 하므로 K는 3으로 고정하고 weights만 바꿉니다.
    # distance의 비중은 거리의 역수입니다. 가장 가까운 입력 4는 거리 0.8이라 정답 40이 더 크게 반영됩니다.
    model = KNeighborsRegressor(n_neighbors=3, weights=weight)
    model.fit(X, y)
    print(weight, "예측:", model.predict([[3.2]]).round(2))
