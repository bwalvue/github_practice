"""작업 복잡도로 성공/실패를 분류하고 확률을 확인합니다."""
from sklearn.linear_model import LogisticRegression

X = [[1], [2], [3], [4], [5], [6]]

# 이 0/1은 크기를 예측할 숫자가 아니라 성공/실패를 구분하는 클래스 라벨입니다.
y = [0, 0, 0, 1, 1, 1]

model = LogisticRegression()

# TODO B03: model을 X와 y로 학습하세요.
# 풀이: LogisticRegression에 0/1 라벨을 학습시켜, 입력 복잡도에 따른 두 클래스의 경계를 구합니다.
model.fit(X, y)

new = [[2], [5]]

# predict_proba의 열은 라벨 번호가 아니라 classes_의 순서입니다.
# 여기서는 [0, 1]이므로 첫 열이 성공 확률, 둘째 열이 실패 확률입니다.
print("클래스 순서:", model.classes_)
print("확률 [성공, 실패]:")
print(model.predict_proba(new).round(3))
print("예측 클래스:", model.predict(new))
