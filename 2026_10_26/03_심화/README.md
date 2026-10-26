# 심화 · 구조는 체계적으로, 내용은 종전 응용 수준

심화라는 이름 때문에 고급 알고리즘·복잡한 예외처리·자동 테스트 작성으로 확장하지 않습니다. 기존 응용의 다중 입력·오류 사례·가중치·전처리 비교·K 탐색·저장 복원을 역할별로 정리합니다. **전체 TODO10곳**이며 모든 단계를 의무 수행하지 않습니다.

이 학습본은 **TODO 10곳의 실행 가능한 정답을 모두 작성**했습니다. 원본 TODO 문구를 그대로 남겼으므로 `TODO H`로 검색하면 풀이 위치를 찾을 수 있습니다. `preprocessing.py`와 `search.py`는 제공된 계산을 유지하고, 비교 조건과 객체를 새로 만드는 이유를 주석으로 보충했습니다.

| 실습 | 들어갈 폴더 | 과제 |
|---|---|---|
| 01 | [실습01_선형회귀](실습01_선형회귀/README.md) | 두 계수·절편과 손계산 |
| 02 | [실습02_회귀오차](실습02_회귀오차/README.md) | 기준선 비교·큰 잔차3건 |
| 03 | [실습03_확률과오류](실습03_확률과오류/README.md) | 혼동행렬·가상 임계값2개 |
| 04 | [실습04_KNN가중치](실습04_KNN가중치/README.md) | 거리 표·가중 평균 |
| 05 | [실습05_스케일링](실습05_스케일링/README.md) | 세 스케일·중앙값 불변 |
| 06 | [실습06_K비교](실습06_K비교/README.md) | 분류F1·회귀RMSE, K 선택 |
| 07~08 | [day6_compare](day6_compare/README.md) | 모듈형 프로젝트·복원·보고서 |

각 폴더의 README부터 읽습니다. `data/`는 입력, `expected/`는 비교 자료, `outputs/`는 직접 생성 결과, `reports/`는 자신의 설명입니다. 위의 순서로 진행하되 각 실습은 독립 실행됩니다. 마지막 프로젝트도 앞 실습 결과 파일을 요구하지 않습니다.

## 완성본 실행과 리뷰

`2026_10_26` 폴더에서 필요한 패키지가 설치된 가상환경을 사용합니다. 실습01~06은 해당 폴더의 `main.py`를 실행합니다. 프로젝트는 학습·저장 후 복원하는 순서입니다.

```bash
python "03_심화/실습01_선형회귀/main.py"
python "03_심화/day6_compare/main.py"
python "03_심화/day6_compare/restore.py"
```

실습01~06과 프로젝트·복원 실행 결과를 `expected/`와 대조했습니다. 생성한 CSV 9개는 열·행 순서·수치가 예상 결과와 일치하며, 별도 Python 실행에서 회귀·분류의 복원 예측이 모두 `True`입니다. 선택 근거와 오류 사례는 [모델 선택 보고서](day6_compare/reports/model_choice.md)에 적었습니다.

리뷰할 때는 TODO 아래의 풀이와 해당 실습의 결과 CSV를 함께 봅니다. 결과 CSV·선택 JSON은 Git에 포함하며 `models.joblib`은 제외합니다. 복원 실습용 모델은 `main.py`를 실행해 만듭니다.

실습01과04는 코드에 작은 입력이 있으므로 빈 data 폴더를 억지로 만들지 않았습니다. 여러 파일로 나누는 것은 책임을 분리하기 위한 것이지, 폴더 수를 늘리는 목표가 아닙니다.

**사용하지 않는 과제:** 거리 동률·과도한 K의 실패 테스트, 입력 계약 검증기, C 추가 탐색, JSON 설정 프레임워크, 그룹/시간 분할, 모델 보안 검사 구현. 제작 검수는 강사가 별도로 수행합니다.

응용·심화의 `requests.csv`는 원본과 같고, 분할 파일은 기존 소속·순서를 합친 단일 표입니다. test 입력·정답은 모델에서 사용하지 않습니다.

## 공식 참고자료

아래 문서는 함수 동작을 더 확인할 때 사용하는 참고 자료이며, 링크를 모두 읽어야 실습을 할 수 있는 것은 아닙니다. 예제 출력은 README에 적은 설치 버전에서 실행해 확인했습니다.

- [선형회귀](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LinearRegression.html)
- [로지스틱회귀](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [최근접 이웃](https://scikit-learn.org/1.8/modules/neighbors.html)
- [표준화](https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [RMSE](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.root_mean_squared_error.html)
- [데이터 누수와 Pipeline](https://scikit-learn.org/stable/common_pitfalls.html)
- [모델 저장·복원 주의점](https://scikit-learn.org/stable/model_persistence.html)
