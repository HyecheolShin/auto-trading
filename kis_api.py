import requests
import json
import time
import os
import urllib3
from datetime import datetime
from dotenv import load_dotenv
from options_pricing import black_scholes_delta

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

APP_KEY    = os.getenv("APP_KEY")
APP_SECRET = os.getenv("APP_SECRET")
ACCOUNT_NO = os.getenv("ACCOUNT_NO")
BASE_URL   = "https://openapivts.koreainvestment.com:29443"
TOKEN_CACHE = ".token_cache.json"
RECORDS_FILE = "trading_records.json"

CALL_INTERVAL = 3  # 호출 간 최소 대기(초)


# ================================================================
# 거래 기록 로그 (모의투자 체결내역 조회가 개별 내역을 채워주지 않아
# 주문 응답을 직접 기록하는 방식으로 대체)
# ================================================================
def log_trade(order_type, ticker, qty, price, response):
    records = []
    if os.path.exists(RECORDS_FILE):
        with open(RECORDS_FILE, "r", encoding="utf-8") as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                records = []

    output = response.get("output", {})
    records.append({
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "order_type":  order_type,
        "ticker":      ticker,
        "qty":         qty,
        "price":       price,
        "order_no":    output.get("ODNO", ""),
        "order_time":  output.get("ORD_TMD", ""),
        "rt_cd":       response.get("rt_cd", ""),
        "msg":         response.get("msg1", "").strip()
    })

    with open(RECORDS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# ================================================================
# 안전 호출 래퍼: 에러 시 자동 대기 후 재시도
# ================================================================
def safe_request(method, url, headers, params=None, data="", retries=3):
    for attempt in range(retries):
        response = requests.request(method, url, headers=headers,
                                    params=params, data=data, verify=False)
        result = response.json()
        code = result.get("rt_cd", "0")
        msg  = result.get("msg_cd", "")

        if code == "0":
            return result  # 성공

        if msg == "EGW00215":  # 초당 거래건수 초과
            wait = 10 * (attempt + 1)
            print(f"  [경고] 호출 속도 초과 - {wait}초 대기 후 재시도 ({attempt+1}/{retries})")
            time.sleep(wait)
            continue

        if msg == "EGW9999":   # 게이트웨이 차단
            print(f"  [오류] 게이트웨이 차단 상태 (EGW9999). 잠시 후 다시 시도하세요.")
            return result

        # 그 외 에러
        print(f"  [에러] {result.get('msg1', '')} ({msg})")
        return result

    print(f"  [실패] {retries}회 재시도 모두 실패")
    return result


# ================================================================
# 1. 토큰 발급 (캐시 재사용)
# ================================================================
def get_access_token():
    if os.path.exists(TOKEN_CACHE):
        with open(TOKEN_CACHE, "r") as f:
            cached = json.load(f)
        expired_str = cached.get("access_token_token_expired", "")
        try:
            expired_dt = datetime.strptime(expired_str, "%Y-%m-%d %H:%M:%S")
            if datetime.now() < expired_dt:
                print(f"[토큰 재사용] 만료: {expired_str}")
                return cached["access_token"]
        except Exception:
            pass

    url = f"{BASE_URL}/oauth2/tokenP"
    payload = json.dumps({
        "grant_type": "client_credentials",
        "appkey":     APP_KEY,
        "appsecret":  APP_SECRET
    })
    response = requests.request("POST", url,
                                headers={"content-type": "application/json"},
                                data=payload, verify=False)
    token_data = response.json()

    if "access_token" not in token_data:
        raise SystemExit(f"[토큰 발급 실패] {token_data}")

    with open(TOKEN_CACHE, "w") as f:
        json.dump(token_data, f)

    print(f"[토큰 발급 완료] 만료: {token_data.get('access_token_token_expired', '')}")
    return token_data["access_token"]


# ================================================================
# 2. 잔고 조회
# ================================================================
def get_balance(access_token):
    url = f"{BASE_URL}/uapi/domestic-stock/v1/trading/inquire-balance"
    headers = {
        "content-type":  "application/json",
        "authorization": f"Bearer {access_token}",
        "appkey":        APP_KEY,
        "appsecret":     APP_SECRET,
        "tr_id":         "VTTC8434R"
    }
    params = {
        "CANO":                   ACCOUNT_NO,
        "ACNT_PRDT_CD":           "01",
        "AFHR_FLPR_YN":           "N",
        "OFL_YN":                 "N",
        "INQR_DVSN":              "02",
        "UNPR_DVSN":              "01",
        "FUND_STTL_ICLD_YN":      "N",
        "FNCG_AMT_AUTO_RDPT_YN":  "N",
        "PRCS_DVSN":              "01",
        "CTX_AREA_FK100":         "",
        "CTX_AREA_NK100":         ""
    }
    data = safe_request("GET", url, headers, params=params)
    output2 = data.get("output2", [{}])
    if output2:
        cash = output2[0].get("dnca_tot_amt", "N/A")
        print(f"  예수금: {cash}원")
    return data


# ================================================================
# 2-1. 현재가 조회
# ================================================================
def get_current_price(access_token, ticker):
    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price"
    headers = {
        "content-type":  "application/json",
        "authorization": f"Bearer {access_token}",
        "appkey":        APP_KEY,
        "appsecret":     APP_SECRET,
        "tr_id":         "FHKST01010100"
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD":         ticker
    }
    data = safe_request("GET", url, headers, params=params)
    price = int(data.get("output", {}).get("stck_prpr", 0))
    print(f"  현재가({ticker}): {price}원")
    return price


def get_holding_qty(access_token, ticker):
    balance = get_balance(access_token)
    for item in balance.get("output1", []):
        if item.get("pdno") == ticker:
            return int(item.get("hldg_qty", 0))
    return 0


# ================================================================
# 2-2. 델타 헤징 (수업의 Black-Scholes 델타 헤징 방식 적용)
#
# 005930 콜옵션을 매도(short call)했다고 가정하고, Black-Scholes 델타에
# 맞춰 기초자산(주식)을 목표 수량만큼 보유하도록 실제 매수/매도 주문을
# 실행한다. (참고: [7_1]_delta_hedging.ipynb, [8_1]_BlackScholes.ipynb)
# ================================================================
def delta_hedge_trade(access_token, ticker="005930", T=30/365, r=0.035,
                       sigma=0.25, contract_qty=10, strike=None):
    spot = get_current_price(access_token, ticker)
    if strike is None:
        strike = spot  # ATM 콜옵션 가정

    delta = black_scholes_delta(S=spot, K=strike, T=T, r=r, sigma=sigma,
                                 option_type="call")
    target_qty = round(delta * contract_qty)
    current_qty = get_holding_qty(access_token, ticker)
    diff = target_qty - current_qty

    print(f"  [델타헤징] spot={spot} strike={strike} delta={delta:.4f} "
          f"목표보유={target_qty} 현재보유={current_qty} 차이={diff}")

    if diff > 0:
        data = place_order(access_token, ticker, order_type="buy", qty=diff, price=0)
    elif diff < 0:
        data = place_order(access_token, ticker, order_type="sell", qty=abs(diff), price=0)
    else:
        print("  [델타헤징] 리밸런싱 불필요 (목표 보유량과 동일)")
        return None

    log_hedge(ticker, spot, strike, delta, target_qty, current_qty, diff, data)
    return data


def log_hedge(ticker, spot, strike, delta, target_qty, current_qty, diff, response):
    records = []
    if os.path.exists(RECORDS_FILE):
        with open(RECORDS_FILE, "r", encoding="utf-8") as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                records = []

    output = response.get("output", {})
    records.append({
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strategy":     "black_scholes_delta_hedge",
        "ticker":       ticker,
        "spot":         spot,
        "strike":       strike,
        "delta":        round(delta, 4),
        "target_qty":   target_qty,
        "current_qty":  current_qty,
        "order_qty":    abs(diff),
        "order_type":   "buy" if diff > 0 else "sell",
        "order_no":     output.get("ODNO", ""),
        "order_time":   output.get("ORD_TMD", ""),
        "rt_cd":        response.get("rt_cd", ""),
        "msg":          response.get("msg1", "").strip()
    })

    with open(RECORDS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# ================================================================
# 3. 주식 주문 (매수 / 매도)
# ================================================================
def place_order(access_token, ticker, order_type="buy", qty=1, price=0):
    url = f"{BASE_URL}/uapi/domestic-stock/v1/trading/order-cash"
    tr_id    = "VTTC0802U" if order_type == "buy" else "VTTC0801U"
    is_market = (price == 0)

    body = {
        "CANO":         ACCOUNT_NO,
        "ACNT_PRDT_CD": "01",
        "PDNO":         ticker,
        "ORD_DVSN":     "01" if is_market else "00",
        "ORD_QTY":      str(qty),
        "ORD_UNPR":     "0" if is_market else str(price)
    }
    headers = {
        "content-type":  "application/json",
        "authorization": f"Bearer {access_token}",
        "appkey":        APP_KEY,
        "appsecret":     APP_SECRET,
        "tr_id":         tr_id
    }
    data = safe_request("POST", url, headers, data=json.dumps(body))
    if data.get("rt_cd") == "0":
        action = "매수" if order_type == "buy" else "매도"
        odno = data.get("output", {}).get("ODNO", "")
        print(f"  {action} 주문 접수 완료 - 주문번호: {odno}")
    return data


# ================================================================
# 실행
# ================================================================
if __name__ == "__main__":
    access_token = get_access_token()

    print("\n" + "=" * 50)
    print(f"[1] 잔고 조회 (헤징 전)")
    get_balance(access_token)
    time.sleep(CALL_INTERVAL)

    print("\n" + "=" * 50)
    print(f"[2] 삼성전자(005930) 콜옵션 매도 가정 - Black-Scholes 델타 헤징")
    delta_hedge_trade(access_token, ticker="005930", T=30/365, r=0.035,
                       sigma=0.25, contract_qty=10)
    time.sleep(CALL_INTERVAL)

    print("\n" + "=" * 50)
    print(f"[3] 잔고 조회 (헤징 후)")
    get_balance(access_token)

    print("\n완료")
