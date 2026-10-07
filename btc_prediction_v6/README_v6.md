# V6: 여러 코인 추세 추종

## 1. 데이터 받기 (본인 PC에서)
```
cd btc_prediction_v6
python download_data.py
```
BTC, ETH, BNB, XRP, ADA, LTC, LINK, TRX, DOGE, SOL의 1시간봉을 `data/{SYMBOL}_1h.csv`로 받는다(약 100MB).
다시 실행하면 이어서 받는다.

## 2. 실행
```
python btc_v6.py
```

## 데이터를 보기 전에 고정한 설계
- 코인별 규칙: 모의매매에 고정한 규칙(`paper_trading/strategy_frozen.py`)을 그대로 쓴다
- **PRIMARY = MULTI_TREND**: 신호가 유효한 코인 N개에 1/N씩 자금을 나누고, 각 칸 안에서 코인별 추세 비중을 적용한다.
  총 비중 ≤ 100%, 레버리지 없음, 현물 롱만
- 비용(편도): BTC/ETH 0.15%, 그 외 0.20%
- 기간과 기준: V5와 같다 (dev 2018-10 ~ 2026-04, 홀드아웃 최근 180일, 기준 5개는 BTC 단순 보유와 비교)
- 비교 대상: BTC 단순 보유, 코인 동일비중 보유(EW_BUY_HOLD), BTC 단독 추세

## 알려진 편향
코인 목록이 "지금까지 살아남은 코인"이라 모든 다중 코인 결과가 실제보다 좋게 나온다(생존 편향).
그래서 MULTI_TREND는 동일비중 보유(EW_BUY_HOLD)보다도 나아야 의미가 있다.
