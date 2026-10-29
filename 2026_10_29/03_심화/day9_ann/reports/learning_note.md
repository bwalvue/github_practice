# ANN 학습 설명

## 파일별 책임
- `main.py`: 고정 분할 로드, 학습·검증 연결, 결과 및 파라미터 저장
- `model.py`: 입력2→은닉8→출력2 구조와 logits 반환
- `train.py`: 한 epoch 학습과 평가 전용 경로 분리
- `check_gpu.py`: CUDA 가용성에 따른 장치 선택 및 실제 환경 기록
- `restore.py`: 새 프로세스의 CPU 모델로 파라미터 복원 및 ID·예측 비교

## 학습 순서와 손실 집계
- `zero_grad`: 이전 배치의 gradient 초기화
- `forward → loss`: 현재 파라미터의 logits와 CrossEntropyLoss 계산
- `backward → step`: 현재 손실의 gradient 계산 후 SGD 파라미터 갱신
- 마지막 batch24건: 각 배치 평균 loss에 실제 행 수를 곱해 합산한 뒤 전체120건으로 나누는 가중 집계
- `train_loss_online`: 갱신 중 서로 다른 파라미터에서 계산한 배치 손실의 표본 가중 평균
- `final_train_loss`: 학습 종료 모델로 train120건을 다시 평가한 고정 모델 손실

## 검증과 한계
- `model.eval()`: 평가용 계층 동작 선택
- `torch.no_grad()`: 평가 연산의 계산 그래프 생성 방지 및 기존 gradient 보존
- 실제로 사용한 device: cuda:0, NVIDIA GeForce RTX 5080 Laptop GPU
- 데이터 사용 범위: train120건 학습, valid40건 평가, test40건은 Tensor 변환·모델 입력 모두 제외
- valid 결과: loss 0.057217, accuracy 1.0 (40/40건 정답); 최종 train 전체 평가 loss 0.089844, 마지막 epoch online loss 0.090861
- 해석의 한계: 단순한 경계 규칙으로 만든 작은 합성 데이터의 결과이며, 실제 서비스의 분포·노이즈·일반화 성능에 대한 증거 부족

## 복원
- 저장 형식: 전체 Python 모델 객체 대신 직접 학습한 `state_dict`
- 복원 조건: 동일 ANN 구조, `map_location="cpu"`, `weights_only=True`, 재학습 없는 `eval()` 및 `no_grad()` 추론
- 새 프로세스 ID 순서·예측 비교: CPU 복원 후 ID 순서 True, 예측 일치 True
- 학습 버전과 장치: Python 3.13.16 / PyTorch 2.14.0+cu130 / CUDA 빌드 13.0 / 학습 cuda:0, 복원 cpu
- 재현성 범위: 같은 seed에도 버전·CPU/GPU가 다르면 실수 연산의 마지막 비트까지 같은 결과를 보장하지 못하는 한계

## 제공 테스트
- H11: lr=0으로 파라미터를 고정한 3·3·1 배치의 가중 평균과 전체 손실을 소수점5자리까지 비교
- H12: 평가 전후 파라미터·기존 gradient의 동일성과 예측의 requires_grad=False 확인
- 실행 결과: 두 테스트 모두 통과 (unittest, Ran 2 tests / OK)

환경 참고: NumPy 미설치 경고 발생, 이번 실습의 Tensor·CSV 경로와 제공 테스트는 정상 완료.
