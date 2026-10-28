"""실습 02: 실패 클래스의 연속 점수를 이용한 AUC 계산"""
from sklearn.metrics import roc_auc_score, roc_curve
# AUC 원리 확인용 가상 예제 4건, 보관 test 결과와 무관
y = [0, 0, 1, 1]
score = [0.10, 0.40, 0.35, 0.80]
# TODO B02: roc_auc_score에 실제값과 연속 점수를 넣으세요.
# 양성·음성의 순위 비교를 위해 0/1 판정 대신 연속 점수 사용
# 점수 이진화에 따른 순위 정보 손실 방지 목적
auc = roc_auc_score(y, score)
# TODO 끝
fpr, tpr, thresholds = roc_curve(y, score)
print("FPR:", fpr.round(2).tolist())
print("TPR:", tpr.round(2).tolist())
print(f"AUC: {auc:.2f}")
print("양성 점수 0.35·0.80과 음성 점수 0.10·0.40을 네 번 비교하세요.")
