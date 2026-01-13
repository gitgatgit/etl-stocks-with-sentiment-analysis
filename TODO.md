# TODO

## ML Model Improvements

### High Volatility Class - Low Precision (39.8%)
The model over-predicts high volatility, causing many false alarms.

**Current metrics:**
| Class  | Precision | Recall | F1    |
|--------|-----------|--------|-------|
| high   | 39.8%     | 74.2%  | 51.8  |
| low    | 84.6%     | 77.3%  | 80.8  |
| medium | 86.7%     | 79.1%  | 82.7  |

**Action items:**
- [ ] Implement class balancing (SMOTE or class weights)
- [ ] Raise prediction threshold for high volatility class
- [ ] Add stricter features for high volatility detection
- [ ] Consider separate binary classifier for high vs not-high
- [ ] Add VIX, earnings dates, news sentiment features
- [ ] Implement cost-sensitive learning to penalize false positives

## Dashboard

- [ ] Create Metabase dashboard with key metrics
- [ ] Add alerting for high volatility predictions
