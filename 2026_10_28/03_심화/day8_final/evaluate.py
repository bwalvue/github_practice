"""정상적인 0/1 두 클래스 입력의 지표 계산 모듈"""
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def summarize(y, prediction, probability):
    """0/1 판정 기반 F1·연속 확률 기반 AUC 계산, 보고서 해석은 별도 판단"""
    # 클래스 [0, 1] 고정으로 TN·FP·FN·TP의 위치와 반환 순서 일치
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    # TODO H07m: F1은 prediction으로, ROC-AUC는 probability로 계산하세요.
    # F1에는 0/1 예측, AUC에는 실패 확률 사용
    # 임시 상수 대신 실제 예측 기반 지표 계산
    f1 = f1_score(y, prediction, zero_division=0)
    auc = roc_auc_score(y, probability)
    # TODO 끝
    return {"n": len(y), "Accuracy": accuracy_score(y, prediction),
            "Precision": precision_score(y, prediction, zero_division=0),
            "Recall": recall_score(y, prediction, zero_division=0), "F1": f1, "ROC_AUC": auc,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "actual_failed": int(sum(y))}
