"""글자 수와 허용 도구 수, 두 입력으로 실행시간을 예측합니다."""
from sklearn.linear_model import LinearRegression

# 열 순서는 [글자 수, 도구 수]입니다. coef_[0]·coef_[1]도 이 순서에 대응하므로 예측 때 유지해야 합니다.
X = [[200, 1], [400, 1], [200, 3], [600, 2]]
y = [700, 800, 900, 1000]

model = LinearRegression()

# TODO A01: 두 입력과 정답으로 model을 학습하세요.
# 풀이: 두 입력을 함께 학습해야 글자 수의 계수 0.5와 도구 수의 계수 100을 따로 구할 수 있습니다.
model.fit(X, y)

print("계수:", model.coef_.round(1))
print("절편:", round(model.intercept_, 1))

# 계수 순서에 맞춰 500 + 0.5*400 + 100*2로 손계산하면, 열을 뒤집어 전달하지 않았는지 확인할 수 있습니다.
print("[400, 2] 예측(ms):", model.predict([[400, 2]]).round(1))
print("손계산(ms):", 500 + 0.5 * 400 + 100 * 2)
