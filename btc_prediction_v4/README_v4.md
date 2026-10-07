# BTC V4 — 포지션 기반 워크포워드 리서치

V1~V3는 "12시간 뒤에 오를까?"를 맞히려 했지만 AUC가 0.5 근처에 머물렀다.
V4는 질문을 "지금 BTC를 얼마나 들고 있어야 하나(0~100%)?"로 바꾸고,
모든 전략을 같은 기준표(scorecard)로, 수수료와 슬리피지를 뺀 out-of-sample 성과로 판정한다.

## 실행

```
pip install -r requirements_v4.txt
python btc_v4.py                        # 캐시된 3년치 데이터 사용
python btc_v4.py --years 9 --refresh    # Binance에서 2017년부터 다시 받기 (권장)
```

## 기간 구분
- warmup: 처음 `--min-train-days`(기본 365일). 특징 계산과 첫 ML 학습에 사용
- dev: 워크포워드 out-of-sample 구간. 버전끼리 비교하고 고를 때는 이 구간만 쓴다
- holdout: 마지막 `--holdout-days`(기본 180일). 확인용으로만 보고하고, 고를 때는 쓰지 않는다

## 전략
| 이름 | 내용 |
|---|---|
| BUY_HOLD | 항상 100% 보유 |
| BH_VOLTARGET | 변동성 목표(연 40%)에 맞춰 비중 축소 |
| TREND | 1주/2주/1개월 모멘텀 투표 × 변동성 스케일 |
| ML | LightGBM 24h 방향 확률 → 비중 (30일마다 재학습, purge 적용) |
| TREND_x_ML | 위 둘을 반씩 |

리밸런싱은 하루 1회(UTC 00시), 비중 변화가 10%p 이하면 매매하지 않는다.

## 실전성 기준 (dev 구간, 전부 통과해야 PASS)
1. Sharpe ≥ 1.0
2. Sharpe > Buy&Hold
3. |MDD| ≤ 0.75 × Buy&Hold MDD
4. 블록 부트스트랩 Sharpe 하위 5% > 0
5. 분기 수익률이 플러스인 비율 ≥ 60%

## 결과 파일 (`results/`)
`summary_v4.csv`, `scorecard_v4.csv`, `trend_grid_v4.csv`, `ml_importance_v4.csv`,
`equity_v4.png`, `latest_signal_v4.json`
