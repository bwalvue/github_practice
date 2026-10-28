"""실습 03: 같은 회귀 문제에서도 주 지표에 따라 후보의 순서가 달라집니다."""
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
y = [100, 200, 300, 400]  # 실제 실행시간(ms)
predictions = {"A": [100, 200, 300, 480], "B": [130, 230, 330, 430]}
for name, pred in predictions.items():
    mae = mean_absolute_error(y, pred)
    # TODO B03: 같은 y·pred로 RMSE를 계산하세요. 함수 이름의 root는 제곱근을 뜻합니다.
    # 정답 예시 B03 (아래 임시 대입문을 다음 한 줄로 교체):
    # rmse = root_mean_squared_error(y, pred)
    # 해설: 반복문이 고른 현재 후보 pred와 같은 정답 y를 비교합니다.
    # RMSE는 오차를 제곱한 값의 평균에 제곱근을 취하므로 결과 단위는 원래와 같은 ms입니다.
    # A의 절대 오차는 [0, 0, 0, 80]이므로 sqrt(6400/4)=40ms입니다.
    # B의 절대 오차는 모두 30이므로 sqrt(3600/4)=30ms입니다.
    # MAE로는 A(20ms), RMSE로는 B(30ms)가 작습니다. 큰 오차를 더 벌주는 정도가 다릅니다.
    rmse = 0.0
    # TODO 끝
    print(f"{name}: MAE={mae:.1f}ms, RMSE={rmse:.1f}ms")
print("실행시간 단위 ms를 함께 적으세요. 교차검증은 06번에서 이어집니다.")
