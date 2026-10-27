"""실습 01 응용: 잎 안 정답의 평균으로 숫자를 예측합니다."""

# AI 작성 참고 풀이: 원본 TODO를 유지하고 실행 코드와 해설을 추가했습니다.

from sklearn.tree import DecisionTreeRegressor

X = [[100], [200], [300], [700], [800], [900]]
y = [100, 120, 140, 700, 720, 740]
model = DecisionTreeRegressor(max_depth=1, random_state=42)
# TODO A01: 회귀 트리를 학습하고 세 입력을 예측하세요.
# 풀이: 회귀의 y는 0/1 라벨이 아니라 예측하려는 숫자입니다.
# 기본 설정의 회귀 트리는 새 입력이 도착한 잎에 속한 정답의 평균을 반환합니다.
model.fit(X, y)
prediction = model.predict([[250], [850], [1200]])
# 기준 500의 왼쪽 평균은 120, 오른쪽 평균은 720입니다.
# 1200도 오른쪽 잎으로 가므로 학습 범위 밖에서 직선처럼 값을 연장하지 않습니다.

print("분기 기준:", model.tree_.threshold[0])
print("250·850·1200 예측:", prediction.tolist())
print("왼쪽 평균:", sum(y[:3]) / 3, "오른쪽 평균:", sum(y[3:]) / 3)
print("1200은 학습 범위 밖입니다. 트리도 기존 잎 값으로 예측합니다.")
