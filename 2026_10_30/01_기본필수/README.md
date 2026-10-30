# 기본필수 07 · 학습률 비교

`07_project.py`와 `points.csv`로 독립 실행. [공통 환경 안내](../README.md).

```bash
python 07_project.py
```

고정 조건: CPU, seed42, 2→8→2 모델, AdamW(weight_decay=0), full-batch, 20epoch. train12·valid4 사용, test4 보관. 변경 조건은 학습률만 0.001 대 0.01.

리뷰 위치: B08 역전파·파라미터 갱신, B09 epoch 종료 후 train·valid 손실 기록.

| 후보 | 학습률 | best epoch | best valid loss |
|---|---:|---:|---:|
| baseline | 0.001 | 20 | 0.480205 |
| candidate | 0.01 | 20 | 0.307332 |

실제 결과: [실험표](experiment_table.csv), [baseline 곡선](curves_baseline.png), [candidate 곡선](curves_candidate.png). 학습 이력은 `history_baseline.csv`, `history_candidate.csv`.

07~08 해석: 이번 조건에서는 candidate의 valid loss가 더 낮게 관측. 두 후보 모두 마지막 epoch가 최저 손실 지점이며 마지막 값과의 차이 0. 관측한 20epoch 범위를 넘어선 최적 epoch로 단정 불가.

동일 설정 재실행에서 CSV 3개의 수치 일치 확인. 작은 합성 valid4건·단일 seed·같은 valid의 반복 비교에 따른 해석 한계. 다음 확인은 test 대신 추가 검증 자료와 여러 seed를 이용한 학습률 효과의 일관성·결과 변동 점검.
