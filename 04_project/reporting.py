"""집계 담당 모듈. 주문 금액 계산은 Order.amount에 맡깁니다."""

def summarize_orders(orders):
    """입력은 Order 객체의 List입니다. 빈 목록은 모두 0으로 집계합니다."""
    # 완료 P_B2: 객체의 속성은 점으로 읽고, 금액은 amount()를 호출해 구합니다.
    total_quantity = 0
    total_amount = 0
    for order in orders:
        total_quantity += order.quantity
        total_amount += order.amount()
    return {
        "order_count": len(orders),
        "total_quantity": total_quantity,
        "total_amount": total_amount,
    }


def group_orders(orders):
    """선택 응용: {상품명: {quantity, amount}}를 반환합니다."""
    # 완료 P_A1: 상품명마다 별도의 수량 합계와 금액 합계를 보관합니다.
    grouped = {}
    for order in orders:
        item = order.item
        if item not in grouped:
            grouped[item] = {"quantity": 0, "amount": 0}
        grouped[item]["quantity"] += order.quantity
        grouped[item]["amount"] += order.amount()
    return grouped
