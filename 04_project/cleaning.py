"""정제 담당 모듈. 파일을 읽거나 화면에 출력하지 않고 값만 돌려줍니다."""
from validation import check_row, reject_record


def clean_orders(rows):
    """기본·응용 프로젝트용. 제공된 정상 주문 5행을 새 List로 반환합니다."""
    # 완료 P_B1: 원본 행을 수정하지 않고, 정리한 값으로 새 딕셔너리를 만듭니다.
    clean = []
    for row in rows:
        unit_price = int(row["unit_price"])
        quantity = int(row["quantity"])
        clean.append({
            "order_id": row["order_id"].strip(),
            "item": row["item"].strip(),
            "unit_price": unit_price,
            "quantity": quantity,
            "amount": unit_price * quantity,
        })
    return clean


def clean_orders_advanced(rows):
    """선택 심화용. (정상 List, 제외 List)를 반환합니다."""
    # 완료 P_D1: 잘못된 첫 행이 뒤의 정상 행까지 막지 않도록 검증부터 합니다.
    clean = []
    rejected = []
    seen_ids = set()
    for index, row in enumerate(rows, start=1):
        checked, reason = check_row(row)
        if reason is not None:
            rejected.append(reject_record(index, row, reason))
            continue

        order_id = checked["order_id"]
        if order_id in seen_ids:
            rejected.append(reject_record(index, row, "중복 ID: 확인 필요"))
            continue

        seen_ids.add(order_id)
        clean.append(checked)
    return clean, rejected
