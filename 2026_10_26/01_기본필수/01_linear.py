"""공부시간으로 점수를 예측하는 선형회귀 예제."""
from sklearn.linear_model import LinearRegression

# scikit-learn은 입력을 (샘플 수, 특성 수)로 받으므로, 특성이 공부시간 하나여도
# [1, 2, ...] 대신 [[1], [2], ...]로 만들어 2차원을 유지합니다.
X = [[1], [2], [3], [4], [5]]
y = [52, 61, 69, 80, 89]

model = LinearRegression()

# TODO B01: model을 X, y로 학습하고, [[6]]의 예측을 prediction에 넣으세요.
# 풀이: fit에 입력과 정답을 함께 주어 기울기·절편을 구한 뒤, predict에는 새 입력만 줍니다.
model.fit(X, y)

# [[6]]은 1명×특성 1개입니다. [6]을 주면 predict가 요구하는 2차원 입력이 아니어서 오류가 납니다.
prediction = model.predict([[6]])

# coef_는 특성별 계수 배열이고 prediction도 요청별 결과 배열이므로, 각각 첫 원소를 꺼냅니다.
print("기울기:", round(model.coef_[0], 1))
print("절편:", round(model.intercept_, 1))
print("6시간 예측:", round(prediction[0], 1))
