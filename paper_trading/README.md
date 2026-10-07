# 실시간 모의매매 (돈은 쓰지 않음)

두 전략을 나란히 기록한다.
| 봇 | 전략 | 고정 파일 |
|---|---|---|
| `paper_trade.py` | BTC 단독 추세 추종 (V5 TREND_ENS_SMOOTH) | `strategy_frozen.py` |
| `paper_trade_multi.py` | 코인 10개 추세 추종 (V6 MULTI_TREND) | `strategy_frozen.py` + `portfolio_frozen.py` |

`run_paper.bat`은 두 봇을 차례로 실행한다.

## BTC 단독

V5의 `TREND_ENS_SMOOTH`를 **2026-10-07에 고정**한 전략이다. 규칙은 `strategy_frozen.py`에 있으며 이후 수정하지 않는다.
매 실행마다 이 파일의 해시를 로그에 남기므로, 수정되면 바로 드러난다.

## 규칙
- 72시간/1주/2주/1개월/2개월 모멘텀이 플러스인 비율 × min(1, 40% / 최근 1주 변동성), 72시간 지수평활
- 하루 한 번, UTC 00시 봉 마감(한국 시간 10:00)에 판단
- 목표 비중과 현재 비중 차이가 20%p를 넘을 때만 매매. 비용은 편도 0.15%로 가정

## 실행
```
pip install pandas numpy requests
python paper_trade.py            # 한국 시간 10시 이후 하루 1번
python paper_trade.py --verify   # 고정 코드가 V5 백테스트와 같은지 확인 (저장소 전체 필요)
```
며칠 빠뜨려도 다음 실행 때 빠진 날을 순서대로 채운다(최대 약 120일).

### 윈도우 작업 스케줄러에 매일 10:15 등록
```
schtasks /Create /SC DAILY /ST 10:15 /TN BTC_Paper /TR "C:\경로\paper_trading\run_paper.bat"
```

## 기록
- `paper_log.csv`: 판단 시각, 가격, 목표 비중, 매매 전후 비중, 모의 자산, 단순 보유 자산, 전략 해시
- `paper_state.json`: 현재 상태

## 평가 계획 (미리 고정)
- 최소 6개월 운영한 뒤, 같은 기간 단순 보유와 비교한다
- 볼 것: 백테스트와 실제 동작이 같은지(매매 시점, 비중), 최대낙폭, 비용 차감 후 수익
- 6개월은 수익률을 통계적으로 판정하기엔 짧다. 1차 목적은 "규칙이 실제로 백테스트대로 동작하는가" 확인이다

## 코인 10개 (V6 MULTI_TREND), 2026-10-07 고정
- 대상: BTC, ETH, BNB, XRP, ADA, LTC, LINK, TRX, DOGE, SOL
- 신호가 유효한 코인 N개에 1/N 칸씩 나누고, 각 칸 안의 비중은 BTC 봇과 같은 규칙을 따른다. 나머지는 현금
- 비용(편도): BTC/ETH 0.15%, 그 외 0.20%. 상장폐지 등으로 데이터를 못 받은 코인은 매도한다
- 기록: `paper_log_multi.csv` (코인별 매매 + `_PORTFOLIO` 줄에 모의 자산 / BTC 단순 보유 / 동일비중 보유)

```
python paper_trade_multi.py                                  # 매일 1번
python paper_trade_multi.py --verify ..\btc_prediction_v6\data   # V6 백테스트와 같은지 확인
```

### 6개월 뒤 평가 (미리 고정)
MULTI_TREND와 BTC 단독을 같은 기간의 BTC 단순 보유, 동일비중 보유와 비교한다.
볼 것: (1) 실제 동작이 백테스트 규칙과 같은지, (2) 최대낙폭, (3) 비용 차감 후 수익률과 Sharpe.
6개월로는 통계적 결론을 내릴 수 없으며, 백테스트와 크게 어긋나는지 보는 것이 1차 목적이다.
