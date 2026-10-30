# 2026-10-30 실습 07

학습 안정화 비교 코드와 실제 실행 결과.

| 단계 | 비교 내용 | 안내 |
|---|---|---|
| 기본필수 | 학습률 0.001 대 0.01 | [README](01_기본필수/README.md) |
| 응용 | SGD·AdamW·Dropout, 최적 상태 복원·클래스별 Recall | [README](02_응용/README.md) |

## 실행

수업에서 검증된 PyTorch 환경 선택 후 이 폴더에서 실행.

```bash
python "01_기본필수/07_project.py"
python "02_응용/07_project.py"
```

검증 환경: Windows / Python 3.13.16 / PyTorch 2.14.0+cu130 / NumPy 2.5.4 / pandas 3.1.0 / Matplotlib 3.11.2. 두 실습의 실제 연산 장치는 CPU.

`requirements.txt`는 PyTorch 외 패키지의 검증 버전. 기존 정상 환경은 유지하고, 추가 패키지가 필요한 경우에만 `python -m pip install -r requirements.txt` 실행. PyTorch는 PC의 CPU/CUDA 환경에 맞게 별도 준비.

리뷰 위치는 코드의 `# TODO`와 `# >>> / # <<<` 사이. 실행 시 CSV·PNG 갱신, test 데이터 미사용.
