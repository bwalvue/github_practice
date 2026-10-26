"""03 파일 기본필수: 정상 주문 5건. 읽기 → 공백/자료형 정제 → 집계 → 저장."""
from pathlib import Path
from storage import read_csv, read_json, save_results

BASE = Path(__file__).resolve().parent


def load_orders(csv_path, json_path):
    """CSV 주문 List와 JSON 주문 List를 순서대로 합칩니다."""
    # 완료 D_B1: 리스트의 +는 CSV 주문 뒤에 JSON 주문을 이어 붙입니다.
    csv_orders = read_csv(csv_path)
    json_orders = read_json(json_path)
    return csv_orders + json_orders


def clean_one(row):
    """정상 입력 1행 → amount가 추가된 새 Dict. 원본은 변경하지 않습니다."""
    # 완료 D_B2: CSV의 숫자 문자열도 계산할 수 있도록 int로 맞춥니다.
    unit_price = int(row["unit_price"])
    quantity = int(row["quantity"])
    return {
        "order_id": row["order_id"].strip(),
        "item": row["item"].strip(),
        "unit_price": unit_price,
        "quantity": quantity,
        "amount": unit_price * quantity,
    }


def make_summary(rows):
    """정제된 List를 입력받아 건수·수량·금액을 담은 Dict를 반환합니다."""
    # 완료 D_B3: 주문 건수는 행의 개수이고, 상품 수량은 각 행의 quantity 합입니다.
    total_quantity = 0
    total_amount = 0
    for row in rows:
        total_quantity += row["quantity"]
        total_amount += row["amount"]
    return {
        "order_count": len(rows),
        "total_quantity": total_quantity,
        "total_amount": total_amount,
    }


def main():
    rows = load_orders(BASE / "data/orders_basic.csv", BASE / "data/orders_basic.json")
    clean = []
    for row in rows:
        clean.append(clean_one(row))
    summary = make_summary(clean)
    save_results(BASE / "outputs/basic", clean, summary)
    print("원본 주문:", len(rows))
    print("정제 주문:", summary["order_count"])
    print("상품 수량:", summary["total_quantity"])
    print("전체 금액:", summary["total_amount"])
    print("저장·재읽기: 확인 완료")
    print("결과: 03_files/outputs/basic")

if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as error:
        print("[아직 작성할 곳]", error)
        print("해당 TODO 하나부터 채우고 다시 실행하세요. 힌트는 실습안내.html에 있습니다.")
        raise SystemExit(2)
