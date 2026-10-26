"""제공 모듈: 전날 학습한 전처리를 한 함수로 재사용합니다. 직접 구현할 TODO는 없습니다."""
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]

def build_model(estimator, scale="standard"):
    """입력4열 → 수치/문자 전처리 → 전달받은 모델. 호출할 때마다 새 객체를 만듭니다."""

    # none에서도 결측 대체를 유지해야 비교 조건이 숫자 스케일링 방식 하나로 제한됩니다.
    scaler = {"none": "passthrough", "standard": StandardScaler(), "minmax": MinMaxScaler()}[scale]
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", scaler)])
    category = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    # FEATURES의 앞 두 열은 숫자, 뒤 두 열은 문자입니다. 열 순서를 바꾸면 이 분기 기준도 함께 바꿔야 합니다.
    pre = ColumnTransformer([("num", numeric, FEATURES[:2]), ("cat", category, FEATURES[2:])])

    # 매 호출 새 객체를 만들고 전처리까지 묶어 반환하므로, 후보끼리 학습 상태를 공유하지 않고 predict가 같은 변환을 재사용합니다.
    return Pipeline([("pre", pre), ("model", estimator)])
