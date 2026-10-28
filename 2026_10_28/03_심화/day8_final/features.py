"""제공 모듈: 실습04의 행별 공식 재사용, 저장 모델 복원에도 필요한 함수"""
import numpy as np

def add_features(X):
    """자기 행의 수치로 두 열 추가, 원본·인덱스·행 순서 보존"""
    out = X.copy()
    out["log_input_chars"] = np.log1p(out["input_chars"])
    out["chars_per_slot"] = out["input_chars"] / (1.0 + out["allowed_tools"])
    return out
