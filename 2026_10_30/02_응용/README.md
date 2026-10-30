# 응용 07 · Optimizer·Dropout 비교

`07_project.py`와 `data/points.csv`로 독립 실행. [공통 환경 안내](../README.md).

```bash
python 07_project.py
```

고정 조건: CPU, 모델 seed42, shuffle seed43, 2→32→32→2 모델, batch32, 40epoch, 학습률0.01, weight_decay=0. train240·valid80 사용, test80 보관.

비교: SGD 대 AdamW의 갱신 규칙, AdamW 대 AdamW+Dropout(p=0.3)의 규제 효과를 각각 확인.

리뷰 위치: A07 역전파·갱신, A08 평가 모드, A09 최적 state_dict의 독립 사본 보관, A10 실제 클래스별 support·correct·Recall.

| 후보 | best epoch | best valid loss | 마지막 valid loss | 마지막−최저 |
|---|---:|---:|---:|---:|
| SGD | 40 | 0.428053 | 0.428053 | 0 |
| AdamW | 38 | 0.016560 | 0.020212 | 0.003652 |
| AdamW+Dropout | 40 | 0.033904 | 0.033904 | 0 |

실제 결과: [비교표](outputs/07_comparison.csv), [SGD 곡선](outputs/07_sgd_curves.png), [AdamW 곡선](outputs/07_adamw_curves.png), [Dropout 곡선](outputs/07_adamw_dropout_curves.png). 후보별 history·예측·클래스 지표는 `outputs/07_*_history.csv`, `07_*_predictions.csv`, `07_*_class_metrics.csv`.

07~08 해석: 이번 조건에서는 AdamW가 최저 valid loss이며 Dropout 추가 시 개선 없음. 마지막 epoch와 구분하여 최저 손실 시점의 가중치를 메모리에서 복원한 뒤 예측·Recall 계산. Recall의 분모는 예측 클래스 수가 아닌 실제 클래스 표본 수. 모델 파일 저장 없이 매 실행 새로 학습.

동일 설정 재실행에서 CSV 10개의 수치 일치 확인. 합성 valid80건·단일 seed·같은 valid의 후보 및 epoch 선택에 따른 해석 한계. 이번 결과만으로 Optimizer 우열이나 Dropout의 일반적 효과를 단정하지 않고, 다음에는 추가 검증 자료와 여러 seed를 이용한 결과 변동 점검.
