"""실습 04: 정상 숫자 두 열의 중요도를 확인하는 작은 원리 예제입니다."""

# AI 작성 참고 풀이: 원본 TODO를 유지하고 실행 코드와 해설을 추가했습니다.

from sklearn.ensemble import RandomForestClassifier

# 각 행은 [글자 수, 허용 도구 수]입니다. 별도 원리 예제이며 requests.csv와 다른 자료입니다.
X = [[100, 1], [200, 2], [300, 1], [700, 2], [800, 1], [900, 2]]
y = [0, 0, 0, 1, 1, 1]
names = ["글자 수", "허용 도구 수"]
model = RandomForestClassifier(n_estimators=20, max_depth=2, random_state=42, n_jobs=1)
model.fit(X, y)
# TODO B04: 학습한 model의 feature_importances_를 읽으세요.
# 풀이: feature_importances_는 fit 뒤 저장되는 배열 속성이므로 () 없이 읽습니다.
# 배열 순서는 X의 열 순서이며, 아래에서 names와 같은 위치끼리 묶습니다.
importance = model.feature_importances_
# 이 값은 불순도 감소에 근거한 상대적 중요도입니다.
# 실패 확률이나 실패의 원인을 나타내는 값은 아닙니다.

for name, value in zip(names, importance):
    print(f"{name}: {value:.3f}")
print("중요도가 높아도 실패의 원인이라고 단정하지 않습니다.")
