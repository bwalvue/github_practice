"""실습 02: AUC에는 0/1 판정이 아니라 실패 클래스의 연속 점수를 넣습니다."""
from sklearn.metrics import roc_auc_score, roc_curve
# 정상적인 원리 예제 4건입니다. 수업용 파일의 test 결과가 아닙니다.
y = [0, 0, 1, 1]
score = [0.10, 0.40, 0.35, 0.80]
# TODO B02: roc_auc_score에 실제값과 연속 점수를 넣으세요.
# 정답 예시 B02 (아래 임시 대입문을 다음 한 줄로 교체):
# auc = roc_auc_score(y, score)
# 해설: y는 실제 성공0·실패1이고, score는 실패일 가능성을 나타내는 연속 점수입니다.
# 양성 0.35는 음성 0.10보다 높고 0.40보다 낮으며, 양성 0.80은 두 음성보다 높습니다.
# 양성·음성 네 쌍 중 세 쌍의 순서가 맞으므로 AUC=3/4=0.75입니다.
# AUC는 점수의 순위를 평가합니다. 점수를 먼저 0/1로 바꾸면 원래 순위 정보가 사라집니다.
# 따라서 0.75를 정확도나 실패 발견 비율 75%로 해석하지 않습니다.
auc = 0.0
# TODO 끝
fpr, tpr, thresholds = roc_curve(y, score)
print("FPR:", fpr.round(2).tolist())
print("TPR:", tpr.round(2).tolist())
print(f"AUC: {auc:.2f}")
print("양성 점수 0.35·0.80과 음성 점수 0.10·0.40을 네 번 비교하세요.")
