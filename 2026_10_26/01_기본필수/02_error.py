"""실제 실행시간과 예측 실행시간의 차이를 두 지표로 계산합니다."""
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# 두 목록의 같은 위치가 같은 요청이어야 하므로, 실제값과 예측값을 따로 정렬하면 안 됩니다.
actual = [100, 200, 300]
prediction = [120, 180, 360]

# TODO B02: 실제값과 예측값을 두 평가 함수에 넣으세요.
# 풀이: 오차를 그대로 평균내면 +20과 -20이 상쇄되므로 MAE는 절댓값을 평균냅니다.
# RMSE는 제곱으로 큰 오차에 더 큰 비중을 주고, 마지막 제곱근으로 단위를 다시 ms로 맞춥니다.
mae = mean_absolute_error(actual, prediction)

rmse = root_mean_squared_error(actual, prediction)

print("MAE(ms):", round(mae, 2))
print("RMSE(ms):", round(rmse, 2))
