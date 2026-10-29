"""학습 한 epoch와 평가를 분리합니다. 빈 데이터·불량 자료는 이번 단계의 과제가 아닙니다."""
import torch

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    # 직전 평가의 eval 모드에서 학습 모드로 전환
    model.train()
    total, count = 0.0, 0
    for xb, yb in loader:
        # 모델과 입력·레이블의 장치 일치
        xb, yb = xb.to(device), yb.to(device)
        # 매 배치의 새로운 gradient 계산을 위한 초기화
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        # loss.backward() 후 optimizer.step()을 호출하세요.
        # 매 batch마다 초기화하고 새로운 gradient로 갱신해야 합니다.
        # >>> H07
        # TODO H07: 위 요구사항을 읽고 이 블록을 완성하세요.
        # 손실의 역전파 후 현재 gradient를 이용한 파라미터 갱신
        loss.backward()
        optimizer.step()
        # <<< H07
        # 갱신 중 측정한 손실에 실제 배치 표본 수를 곱한 누적
        total += loss.item() * len(yb)
        count += len(yb)
    # 마지막 작은 batch도 실제 샘플 수만큼 반영합니다.
    return total / count


def evaluate(model, X, y, loss_fn, device):
    # 평가용 동작 선택과 계산 그래프 생성 방지의 역할 분리
    model.eval()
    with torch.no_grad():
        logits = model(X.to(device))
        loss = loss_fn(logits, y.to(device)).item()
        # logits에서 클래스 인덱스를 구하고 CPU로 옮겨 prediction에 저장하세요.
        # 평가에서는 backward와 optimizer.step을 호출하지 않습니다.
        # >>> H08
        # TODO H08: 위 요구사항을 읽고 이 블록을 완성하세요.
        # 각 행의 최대 logits 인덱스 선택 후 CPU에서 결과 비교
        prediction = logits.argmax(dim=1).cpu()
        # <<< H08
    return loss, (prediction == y.cpu()).float().mean().item(), prediction
