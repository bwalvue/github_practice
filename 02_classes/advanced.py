"""02 클래스 심화(선택): 상속 대신 객체를 List에 담아 협력시킵니다."""

class Order:
    def __init__(self, item, unit_price, quantity):
        self.item = item
        self.unit_price = unit_price
        self.quantity = quantity

    def amount(self):
        return self.unit_price * self.quantity


class Cart:
    def __init__(self):
        # 완료 C_D1: 객체를 만들 때마다 새로운 리스트를 만듭니다.
        self.orders = []

    def add(self, order):
        """입력은 이미 만든 Order 객체입니다. 문자열이나 Dict가 아닙니다."""
        # 완료 C_D2: 전달받은 Order 객체 자체를 목록에 담습니다.
        self.orders.append(order)

    def total_amount(self):
        # 완료 C_D3: 각 주문의 amount()를 호출해 현재 금액을 더합니다.
        total = 0
        for order in self.orders:
            total += order.amount()
        return total


def main():
    first = Cart()
    second = Cart()
    first.add(Order("아메리카노", 3000, 2))
    first.add(Order("쿠키", 2000, 3))
    print("첫 장바구니 금액:", first.total_amount())
    print("둘째 장바구니 금액:", second.total_amount())
    print("서로 다른 목록:", first.orders is not second.orders)

if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as error:
        print("[아직 작성할 곳]", error)
        print("해당 TODO 하나부터 채우고 다시 실행하세요. 힌트는 실습안내.html에 있습니다.")
        raise SystemExit(2)
