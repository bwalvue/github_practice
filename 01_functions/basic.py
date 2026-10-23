"""01 함수 기본필수: 입력 → 계산 → return → 함수 재사용. 세 TODO만 완성합니다."""

def calculate_amount(price, quantity):
    """입력: 정수 단가·수량. 반환: 정수 금액. 이 단계는 정상 입력만 받습니다."""
    # 완료 F_B1: 계산 결과를 호출한 곳에 돌려줍니다.
    return price * quantity


def receipt_message(item, amount):
    """예: ("아메리카노", 6000) → "아메리카노: 6000원". 출력하지 말고 반환합니다."""
    # 완료 F_B2: f 문자열의 중괄호 안에 변수의 값이 들어갑니다.
    return f"{item}: {amount}원"


def describe_order(item, price, quantity):
    """calculate_amount와 receipt_message를 연결합니다."""
    # 완료 F_B3: 첫 함수의 반환값을 다음 함수의 입력으로 전달합니다.
    amount = calculate_amount(price, quantity)
    return receipt_message(item, amount)


def main():
    # 실행 전 아래 세 줄에 어떤 값이 나올지 적어 봅니다.
    print("금액:", calculate_amount(3000, 2))
    print("문장:", receipt_message("쿠키", 6000))
    print("함수 연결:", describe_order("카페라떼", 4000, 2))

if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as error:
        print("[아직 작성할 곳]", error)
        print("해당 TODO 하나부터 채우고 다시 실행하세요. 힌트는 실습안내.html에 있습니다.")
        raise SystemExit(2)
