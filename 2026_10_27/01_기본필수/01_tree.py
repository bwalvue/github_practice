"""실습 01: 입력을 질문으로 나누는 트리를 만듭니다. 성능 평가가 아닌 원리 예제입니다."""

# AI 작성 참고 풀이: 원본 TODO를 유지하고 실행 코드와 해설을 추가했습니다.

from sklearn.tree import DecisionTreeClassifier

X = [[100], [200], [300], [700], [800], [900]]
y = [0, 0, 0, 1, 1, 1]  # 같은 순서의 정답: 성공 0, 실패 1
model = DecisionTreeClassifier(max_depth=1, random_state=42)
# TODO B01a: X와 y로 model을 학습하세요.
# 풀이: fit은 입력 X와 같은 순서의 정답 y로 분기 기준을 학습합니다.
# X는 요청 6건 × 글자 수 1열입니다. 모델 생성만으로는 tree_가 생기지 않습니다.
model.fit(X, y)

# TODO B01b: 250과 850을 예측하여 prediction에 넣으세요.
# 풀이: predict에는 정답 없이 새 입력만 전달합니다.
# [[250], [850]]은 요청 2건 × 특성 1개의 2차원 입력입니다.
# 두 값은 학습한 분기 기준 500의 서로 다른 쪽으로 이동합니다.
prediction = model.predict([[250], [850]])

print("분기 기준:", model.tree_.threshold[0])
print("250·850 예측:", prediction.tolist())
