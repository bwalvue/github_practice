"""02 클래스 응용: 수량 변경과 Dict 변환. 앞 실습과 독립적으로 실행됩니다."""

class Order:
    def __init__(self, item, unit_price, quantity):
        # 생성 시 입력은 정상이라고 가정합니다. 변경 메서드의 검증을 연습합니다.
        self.item = item
        self.unit_price = unit_price
        self.quantity = quantity

    def amount(self):
        return self.unit_price * self.quantity

    def set_quantity(self, quantity):
        """양의 정수만 허용합니다. 실패하면 ValueError, 원래 수량은 유지합니다."""
        # 완료 C_A1: 검사를 통과하기 전에는 기존 self.quantity를 바꾸지 않습니다.
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("수량은 1 이상의 정수여야 합니다.")
        self.quantity = quantity

    def to_dict(self):
        """직렬화할 수 있는 기본 자료형으로 내용을 꺼냅니다."""
        # 완료 C_A2: 새 딕셔너리의 값을 바꾸어도 객체의 속성은 그대로입니다.
        return {
            "item": self.item,
            "unit_price": self.unit_price,
            "quantity": self.quantity,
            "amount": self.amount(),
        }


def main():
    order = Order("쿠키", 2000, 3)
    order.set_quantity(4)
    print("변경 후:", order.to_dict())
    try:
        order.set_quantity(0)
    except ValueError as error:
        print("입력 안내:", error)
    print("기존 수량 유지:", order.quantity)

if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as error:
        print("[아직 작성할 곳]", error)
        print("해당 TODO 하나부터 채우고 다시 실행하세요. 힌트는 실습안내.html에 있습니다.")
        raise SystemExit(2)
