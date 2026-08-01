# Results -- run: 20260731_143558

## Benchmark Results

|                           |     MAE |     RMSE |   wMAPE (%) |   Params (K) |   Train (s) |
|:--------------------------|--------:|---------:|------------:|-------------:|------------:|
| Each-Grid LSTM (Small)    | 46.5244 |  87.7979 |      3.4400 |     369.9000 |    157.4000 |
| Each-Grid LSTM (Base)     | 46.4340 |  87.6159 |      3.4300 |    1353.9000 |    133.0000 |
| Each-Grid LSTM (Large)    | 46.3836 |  86.2735 |      3.4300 |    5165.1000 |     96.3000 |
| All-Grid LSTM (Small)     | 52.3424 | 105.9738 |      3.8700 |       1.2000 |     58.0000 |
| All-Grid LSTM (Base)      | 51.3948 | 104.0770 |      3.8000 |       4.5000 |     63.6000 |
| All-Grid LSTM (Large)     | 50.4033 | 100.7106 |      3.7300 |      17.2000 |     74.1000 |
| All-Grid DLinear (Small)  | 59.6337 | 126.5307 |      4.4100 |       0.1000 |     13.7000 |
| All-Grid DLinear (Base)   | 59.3968 | 126.1911 |      4.3900 |       0.1000 |     11.0000 |
| All-Grid DLinear (Large)  | 59.9354 | 126.9480 |      4.4300 |       0.1000 |      8.8000 |
| All-Grid PatchTST (Base)  | 53.8227 | 107.5003 |      3.9800 |      25.7000 |     75.0000 |
| All-Grid PatchTST (Large) | 50.7747 | 101.8634 |      3.7600 |     200.6000 |    199.7000 |
| All-Grid Reformer (Base)  | 52.7106 | 106.0916 |      3.9000 |      23.4000 |    909.9000 |
| All-Grid Reformer (Large) | 51.8044 | 103.7816 |      3.8300 |     183.2000 |   1223.1000 |
| R-Treeformer (Base)       | 39.8813 |  71.9840 |      2.9500 |      40.0000 |     95.0000 |
| R-Treeformer (Large)      | 38.8215 |  70.4882 |      2.8700 |     253.1000 |    120.1000 |

## Ablation Study

|                      |     MAE |    RMSE |   wMAPE (%) |   Params (K) |   Train (s) |
|:---------------------|--------:|--------:|------------:|-------------:|------------:|
| Full                 | 37.7989 | 69.4053 |      2.8000 |     253.1000 |    184.1000 |
| w/o RevIN            | 39.1514 | 73.1288 |      2.9000 |     253.1000 |    135.8000 |
| w/o Time Embedding   | 39.0000 | 70.4392 |      2.8800 |     253.1000 |    190.6000 |
| w/o Spatial Position | 44.1739 | 80.7104 |      3.2700 |     253.1000 |     72.5000 |
| Random Mask          | 40.6019 | 73.1658 |      3.0000 |     253.1000 |     74.7000 |

## Adaptation Results

|                             |     MAE |     RMSE |   wMAPE (%) |   Params (K) |   Train (s) |
|:----------------------------|--------:|---------:|------------:|-------------:|------------:|
| R-Treeformer Zero-Shot      | 68.8240 | 147.6303 |      4.9200 |          nan |    nan      |
| R-Treeformer Fine-Tune      | 41.9086 |  77.7353 |      3.0000 |          nan |     46.7000 |
| Each-Grid LSTM Zero-Shot    | 72.0528 | 139.9510 |      5.1500 |          nan |    nan      |
| Each-Grid LSTM Retrain      | 51.1002 |  95.2352 |      3.6500 |          nan |    112.4000 |
| All-Grid LSTM Zero-Shot     | 52.7106 | 103.5548 |      3.7700 |          nan |    nan      |
| All-Grid LSTM Retrain       | 54.8450 | 107.2134 |      3.9200 |          nan |     65.9000 |
| All-Grid DLinear Zero-Shot  | 61.8441 | 123.5900 |      4.4200 |          nan |    nan      |
| All-Grid DLinear Retrain    | 64.2130 | 127.2368 |      4.5900 |          nan |     12.0000 |
| All-Grid PatchTST Zero-Shot | 53.4291 | 106.0489 |      3.8200 |          nan |    nan      |
| All-Grid PatchTST Retrain   | 56.1525 | 109.8102 |      4.0200 |          nan |    173.7000 |
| All-Grid Reformer Zero-Shot | 53.4067 | 106.2857 |      3.8200 |          nan |    nan      |
| All-Grid Reformer Retrain   | 56.1908 | 110.9116 |      4.0200 |          nan |   1258.1000 |
