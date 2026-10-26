"""같은 이웃 3개를 이용해 분류는 투표, 회귀는 평균으로 예측합니다."""
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

X = [[1], [2], [4], [7], [9]]
class_y = [0, 0, 1, 1, 1]
number_y = [10, 20, 40, 70, 90]

# TODO B04a: 분류에서 이웃 수를 3으로 바꾸세요.
# 풀이: 3.2의 이웃은 입력 4, 2, 1이고 라벨은 1, 0, 0이므로 세 이웃의 투표 결과는 0입니다.
# K=1을 그대로 두면 가장 가까운 입력 4의 라벨 1만 사용해서 결과가 달라집니다.
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X, class_y)
print("분류:", classifier.predict([[3.2]]))

# TODO B04b: 회귀에서도 이웃 수를 3으로 바꾸세요.
# 풀이: 같은 세 이웃의 숫자 정답 40, 20, 10을 평균내야 하므로 회귀에도 K=3을 적용합니다.
regressor = KNeighborsRegressor(n_neighbors=3)
regressor.fit(X, number_y)
print("회귀:", regressor.predict([[3.2]]).round(2))
