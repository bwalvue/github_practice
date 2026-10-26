"""실습 01 심화: 기존 응용의 두 입력 선형회귀를 수식과 연결합니다."""
from sklearn.linear_model import LinearRegression

X = [[200, 1], [400, 1], [200, 3], [600, 2]]
y = [700, 800, 900, 1000]
model = LinearRegression()

# TODO H01: 두 특성을 가진 X와 y로 model을 학습하세요.
# 풀이: X의 두 열을 함께 학습해야 글자 수와 도구 수의 계수를 각각 구할 수 있습니다.
model.fit(X, y)
new = [[400, 2]]
prediction = model.predict(new)[0]

# coef_의 순서는 X의 열 순서와 같습니다. 같은 순서로 손계산해 predict의 입력 열을 뒤집지 않았는지 확인합니다.
manual = model.intercept_ + model.coef_[0] * new[0][0] + model.coef_[1] * new[0][1]
print("계수:", model.coef_.round(1))
print("절편:", round(model.intercept_, 1))
print("예측(ms):", round(prediction, 1))
print("손계산(ms):", round(manual, 1))
print("차이:", round(abs(prediction - manual), 8))
