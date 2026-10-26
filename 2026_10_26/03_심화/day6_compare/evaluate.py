"""실습07에서 main.py가 가져다 쓰는 작은 평가 모듈입니다."""
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def regression_metrics(actual, prediction):
    """회귀 오차의 단위는 ms, R²에는 단위가 없습니다."""

    # TODO H07r: MAE·RMSE를 계산하세요.
    # 풀이: 같은 실제값·예측값으로 두 오차를 계산합니다. MAE는 절댓값, RMSE는 제곱으로 큰 오차의 비중을 달리합니다.
    mae = mean_absolute_error(actual, prediction)
    rmse = root_mean_squared_error(actual, prediction)

    # R²의 0 기준은 valid 정답의 평균입니다. train 평균을 쓰는 Dummy의 R²도 음수가 될 수 있습니다.
    return {"MAE_ms": mae, "RMSE_ms": rmse, "R2": r2_score(actual, prediction)}

def classification_metrics(actual, prediction):
    """실패1을 양성으로 고정합니다. zero_division=0은 분모가 0일 때 0점 표시 규칙입니다."""

    # labels=[0, 1]로 순서를 고정해야 펼친 네 값을 TN·FP·FN·TP에 일관되게 대응시킬 수 있습니다.
    tn, fp, fn, tp = confusion_matrix(actual, prediction, labels=[0, 1]).ravel()

    # TODO H07c: 실패1의 F1을 계산하세요.
    # 풀이: 성공만 예측해도 높아질 수 있는 정확도와 구분해, 실패 1을 찾는 정밀도·재현율을 F1로 함께 비교합니다.
    f1 = f1_score(actual, prediction, zero_division=0)
    return {"Accuracy": accuracy_score(actual, prediction), "F1": f1,
            "Precision": precision_score(actual, prediction, zero_division=0),
            "Recall": recall_score(actual, prediction, zero_division=0),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
