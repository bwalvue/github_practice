"""실습 03: 주 지표에 따른 회귀 후보의 순위 비교"""
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
y = [100, 200, 300, 400]  # 실제 실행시간(ms)
predictions = {"A": [100, 200, 300, 480], "B": [130, 230, 330, 430]}
for name, pred in predictions.items():
    mae = mean_absolute_error(y, pred)
    # TODO B03: 같은 y·pred로 RMSE를 계산하세요. 함수 이름의 root는 제곱근을 뜻합니다.
    # 현재 후보와 같은 실제값으로 RMSE 계산, 결과 단위는 ms
    # 큰 오차에 더 큰 영향을 주는 제곱 오차 기반 지표
    rmse = root_mean_squared_error(y, pred)
    # TODO 끝
    print(f"{name}: MAE={mae:.1f}ms, RMSE={rmse:.1f}ms")
print("실행시간 단위 ms를 함께 적으세요. 교차검증은 06번에서 이어집니다.")
