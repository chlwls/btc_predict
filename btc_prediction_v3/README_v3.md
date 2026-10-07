# BTC Prediction V3

V3는 V2의 가장 중요한 문제였던 백테스트 계산 방식을 수정하고, 단순한 방향 예측을 넘어
시장 국면(regime), 멀티 타임프레임, 해석 가능한 패턴 통계, 분류 + 회귀, 워크포워드 검증을 포함한 연구용 BTC 예측 시스템이다.

## 핵심 변경점

1. **백테스트 수정**
   - V2의 겹치는 12시간 forward return을 누적해서 Buy & Hold를 계산하던 문제 제거
   - Buy & Hold = 마지막 종가 / 첫 종가 - 1
   - 한 번 진입하면 horizon 동안 보유하여 중복 포지션 방지
   - 수수료/슬리피지 명시적 반영

2. **멀티 타임프레임**
   - 1시간 실행 데이터
   - 4시간 추세
   - 일봉 추세
   - 4h/1d EMA, RSI, 변동성, 추세 정렬 등을 1h 시점에 결합

3. **시장 국면(Regime)**
   - 1d EMA50/EMA200 관계
   - 4h EMA20/EMA50 관계
   - 변동성 분위수
   - TREND_UP / TREND_DOWN / RANGE_HIGH_VOL / RANGE_LOW_VOL 등으로 분류

4. **패턴 통계**
   - RSI 과매도/과매수
   - EMA 정렬
   - breakout
   - volume spike
   - 큰 양봉/음봉
   - trend alignment
   - 각 패턴의 표본 수, 평균 미래수익률, 승률, expectancy를 CSV로 저장

5. **모델**
   - LightGBM
   - XGBoost
   - HistGradientBoosting
   - 회귀 모델: HistGradientBoostingRegressor
   - validation에서만 threshold 선택
   - test는 완전 out-of-sample 평가

6. **워크포워드**
   - expanding-window
   - 각 구간은 미래 데이터를 학습에 사용하지 않음
   - 구간별 AUC / sign accuracy / return / MDD / trades 기록

## 설치

```powershell
cd C:\Users\M\Desktop\btc_test\btc_prediction_v3
python -m pip install -r requirements_v3.txt
```

## 실행

처음에는 다음 명령을 권장한다.

```powershell
python btc_model_v3.py --years 3 --horizon 12
```

빠른 테스트:

```powershell
python btc_model_v3.py --years 1 --horizon 12
```

비용을 직접 지정할 수도 있다.

```powershell
python btc_model_v3.py --years 3 --horizon 12 --fee 0.001 --slippage 0.0005
```

## 생성 파일

- `data/btc_usdt_1h_raw.csv`
- `data/btc_features_v3.csv`
- `results/model_comparison_v3.csv`
- `results/pattern_statistics_v3.csv`
- `results/walk_forward_v3.csv`
- `results/test_predictions_v3.csv`
- `results/backtest_trades_v3.csv`
- `results/latest_prediction_v3.json`
- `results/equity_curve_v3.csv`
- `results/equity_curve_v3.png`
- `models/best_classifier_v3.joblib`
- `models/best_regressor_v3.joblib`

## 결과 해석

AUC가 0.50 근처라면 방향 예측력이 거의 없다는 뜻이다.
백테스트 수익률이 양수더라도 표본 수가 너무 적거나 특정 구간에만 의존하면 신뢰하기 어렵다.

특히 V3에서는 다음을 같이 봐야 한다.

- TEST AUC
- TEST sign accuracy
- TEST strategy return
- Buy & Hold return
- Maximum Drawdown
- trade count
- win rate
- 패턴별 sample_count와 expectancy
- walk-forward 구간별 성능 일관성

이 프로젝트는 투자 자문이나 수익 보장 시스템이 아니라 시계열 ML 연구/백테스트용이다.
