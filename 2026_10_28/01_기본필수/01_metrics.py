"""실습 01: 실제 실패 6건인 손계산 예제입니다. 아직 모델을 학습하지 않습니다."""
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# 행은 실제, 열은 예측입니다. 1은 좋은 결과가 아니라 관심 클래스인 '실패'입니다.
y = [0]*14 + [1]*6
pred = [0]*10 + [1]*4 + [0]*4 + [1]*2
print("혼동행렬(행 실제 / 열 예측, 성공0·실패1)")
print(confusion_matrix(y, pred, labels=[0, 1]))
# TODO B01: 네 지표 함수에 실제값 y와 예측값 pred를 순서대로 넣으세요. zero_division=0은 분모가 0일 때 0으로 표시하는 정책입니다.
# 정답 예시 B01 (아래 임시 대입문을 다음 네 줄로 교체):
# acc = accuracy_score(y, pred)
# precision = precision_score(y, pred, zero_division=0)
# recall = recall_score(y, pred, zero_division=0)
# f1 = f1_score(y, pred, zero_division=0)
# 해설: 첫 인수 y는 실제 정답, 두 번째 pred는 예측입니다. 관심 클래스는 실패 1입니다.
# 혼동행렬은 [[TN, FP], [FN, TP]]이므로 TN=10, FP=4, FN=4, TP=2입니다.
# Accuracy는 (TN+TP)/전체 = 12/20 = 0.600입니다.
# Precision은 TP/(TP+FP), Recall은 TP/(TP+FN)이며 여기서는 둘 다 2/6입니다.
# F1은 2*TP/(2*TP+FP+FN) = 4/12로, 세 지표 모두 약 0.333입니다.
# zero_division=0은 해당 지표의 분모가 0이면 0으로 처리하며, Accuracy에는 필요하지 않습니다.
acc = precision = recall = f1 = 0.0
# TODO 끝
print(f"Accuracy={acc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
print("실제 실패 6건 중 발견: 2건")
