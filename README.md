# KIS Open API 자동매매 시스템

한국투자증권 Open API를 활용한 Black-Scholes 델타 헤징 기반 자동매매 시스템입니다.

## 주요 파일

| 파일 | 설명 |
|------|------|
| `kis_api.py` | KIS 모의투자 API 연동 및 자동 주문 실행 |
| `options_pricing.py` | 옵션 가격 계산 및 Deep Hedging 구현 |
| `trading_records.json` | 실제 모의투자 체결 기록 |

## 거래 전략

삼성전자(005930) 콜옵션을 매도(short call)했다고 가정하고, Black-Scholes 모델의 델타(Δ)를 계산하여 기초자산을 목표 수량만큼 자동으로 매수/매도합니다.

```
delta = N(d1)  # Black-Scholes 델타
target_qty = round(delta × contract_qty)
→ 현재 보유량과의 차이만큼 자동 주문
```

## 옵션 가격 모델

- **Black-Scholes**: 유럽형 콜/풋 옵션 가격 및 델타
- **Lookback Option**: 몬테카를로 시뮬레이션
- **European Binary**: 해석적 공식
- **American Binary**: 몬테카를로 시뮬레이션
- **Deep Hedging**: MLP 기반 동적 헤징 (entropic risk measure 최소화)

## 실행 방법

```bash
pip install requests python-dotenv scipy numpy torch
```

`.env` 파일에 KIS API 키 설정:
```
APP_KEY=발급받은_앱키
APP_SECRET=발급받은_시크릿
ACCOUNT_NO=계좌번호
```

```bash
python kis_api.py
```

## 실행 결과

`trading_records.json`에 체결 기록이 저장됩니다.

```json
{
  "timestamp": "2026-06-17 12:06:53",
  "strategy": "black_scholes_delta_hedge",
  "ticker": "005930",
  "spot": 336750,
  "delta": 0.5303,
  "target_qty": 5,
  "order_qty": 5,
  "order_type": "buy",
  "order_no": "0000024431",
  "rt_cd": "0",
  "msg": "모의투자 매수주문이 완료 되었습니다."
}
```
