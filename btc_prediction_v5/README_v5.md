# BTC V5: 매매 횟수를 줄인 추세 추종

실행: `python btc_prediction_v5/btc_v5.py` (`btc_prediction_v4/data`의 9년치 1시간봉 사용)

## 실행 전에 고정한 것
- ML 방향 예측 제외 (V1~V4에서 수수료를 빼면 수익이 없었음)
- **PRIMARY = TREND_ENS**: 72h/1주/2주/1개월/2개월 모멘텀 투표를 똑같이 섞고,
  변동성 목표 40%, 하루 1회 리밸런싱, 매매 기준폭 20%p
- 변형(SMOOTH, WEEKLY)은 민감도 확인용이고, 판정은 PRIMARY로만 함
- 5번 기준 변경: "분기 플러스 60%" → "손실 분기 수 < Buy&Hold"
  (V4 결과를 본 뒤 사용자가 결정한 변경임을 기록)

## 결과 (dev 2018-10 ~ 2026-04)
| 전략 | Sharpe | MDD | 손실 분기 | 판정 |
|---|---|---|---|---|
| BUY_HOLD | 0.82 | -76.6% | 14/31 | - |
| TREND_ENS (PRIMARY) | 0.97 | -48.7% | 18/31 | FAIL (3/5) |
| TREND_ENS_SMOOTH | 1.09 | -43.3% | 14/31 | FAIL (4/5) |
| TREND_ENS_WEEKLY | 1.15 | -49.4% | 16/31 | FAIL (4/5), 홀드아웃 -4% |

결과 파일: `results/` (summary, scorecard, yearly, cost_stress, equity_v5.png, latest_signal)
