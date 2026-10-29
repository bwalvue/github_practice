# 2026-10-29 실습 풀이

01~03 전체 자료와 TODO 31곳 풀이. 04_특심화 제외.

| 단계 | TODO | 실행 안내 |
|---|---:|---|
| 기본필수 | 9곳 | [README](01_기본필수/README.md) |
| 응용 | 10곳 | [README](02_응용/README.md) |
| 심화 | 12곳 | [README](03_심화/README.md) |

리뷰 위치: `# TODO B/A/H...`와 `# >>> / <<<` 사이. 주요 흐름에 명사형 설명 주석 추가.

기본필수 CPU, 응용·심화 CUDA 실행. 최종 valid 정확도 모두 1.0, test 미사용. 기본필수·응용 재실행 일치, 심화 테스트 2개 통과 및 새 CPU 프로세스 복원 예측 일치 확인.

심화 실행 순서: `python main.py` → `python restore.py`. 테스트: `python -m unittest discover -s tests -v`.

[심화 학습 기록](03_심화/day9_ann/reports/learning_note.md)에 실제 환경·손실·복원 결과 기재. 모델 가중치는 `main.py` 실행으로 생성.
