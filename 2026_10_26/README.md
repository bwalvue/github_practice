# 2026-10-26 · 회귀·분류 모델 실습

강사님이 제공한 코드와 데이터를 바탕으로 기본필수·응용의 TODO를 완성한 학습 기록입니다. 각 Python 파일은 독립적으로 실행할 수 있습니다. 원본 TODO 주석 바로 아래에 풀이를 두고, 주석에는 해당 입력 모양·전처리·모델 설정을 선택한 이유를 적었습니다.

| 폴더 | 내용 | 입력 데이터 | 결과 저장 위치 |
|---|---|---|---|
| [01_기본필수](01_기본필수/README.md) | 작은 예제로 선형회귀·로지스틱회귀·KNN·표준화 확인 | 폴더 안의 `requests.csv` 및 코드에 있는 예제 | 07번 실행 시 같은 폴더에 CSV 3개 |
| [02_응용](02_응용/README.md) | 요청 100건과 고정 분할로 모델·스케일링·K 비교 | `02_응용/data/` | 실행한 파일 옆의 `outputs/` |

## 실행 환경

Python 3.13.x와 [requirements.txt](requirements.txt)의 버전을 사용합니다. 처음 실행한다면 이 날짜 폴더에서 Git Bash를 열고 다음을 실행합니다. 이미 필요한 패키지가 설치된 가상환경이 있다면 그 환경을 사용해도 됩니다.

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

VS Code의 실행 버튼을 사용할 때도 같은 가상환경의 Python을 선택합니다. 파일은 다음처럼 직접 실행할 수 있습니다.

```bash
python "01_기본필수/01_linear.py"
python "01_기본필수/07_project.py"
python "02_응용/01_linear.py"
python "02_응용/07_project.py"
```

CSV 경로는 각 Python 파일의 위치를 기준으로 계산하므로 위와 같이 날짜 폴더에서 실행해도 됩니다. `outputs/`는 실행할 때 자동으로 만들어집니다. 응용 02번과 07번은 같은 회귀 비교 CSV 이름을 사용하므로 07번 실행 후에는 세 후보의 비교표가 남습니다.

## 결과를 읽는 기준

- 회귀 정답은 `elapsed_ms`입니다. 기본필수의 주 지표는 MAE, 응용의 주 지표는 RMSE이며 둘 다 낮을수록 좋습니다.
- 분류 정답은 `failed`이며 성공은 0, 실패는 1입니다. 실패 1에 대한 F1이 높을수록 좋습니다.
- 전처리 기준과 모델은 train으로 학습하고 valid로 비교합니다.
- 응용 07번은 RMSE로 회귀 후보를, F1으로 분류 후보를 비교합니다. 표에서 선택되는 모델은 각각 Linear와 KNN3입니다.

원본: [강사님 practice_assignment / 20261026](https://github.com/kangminjun2024/practice_assignment/tree/main/20261026). 원본 자료의 라이선스는 [LICENSE](LICENSE)에 보관했습니다.
