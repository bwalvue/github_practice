"""학습 자료에서 구한 평균과 표준편차를 검증 자료에도 적용합니다."""
from sklearn.preprocessing import StandardScaler

train = [[100], [200], [300]]
valid = [[400]]

scaler = StandardScaler()
scaler.fit(train)

# TODO B05: valid를 transform하여 converted에 넣으세요.
# 풀이: fit_transform(valid)를 쓰면 평균이 400으로 덮어써지고, 검증값 400 자체가 0이 됩니다.
# transform만 호출해야 train에서 구한 평균 200과 표준편차를 유지한 채 같은 기준으로 변환합니다.
converted = scaler.transform(valid)

print("train 기준 평균:", scaler.mean_)
print("train 변환:", scaler.transform(train).round(3))
print("valid 변환:", converted.round(3))
print("변환 후 기준 평균:", scaler.mean_)
