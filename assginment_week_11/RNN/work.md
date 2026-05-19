# Báo cáo mô hình RNN dự đoán giá Bitcoin

## 1. Mục tiêu bài toán

Bài toán này xây dựng mô hình RNN để dự đoán giá đóng cửa Bitcoin của ngày tiếp theo. Dữ liệu ban đầu là dữ liệu giao dịch Bitcoin theo từng phút, gồm các cột OHLCV:

- `Timestamp`: thời gian ở dạng Unix timestamp.
- `Open`: giá mở cửa tại mỗi mốc thời gian.
- `High`: giá cao nhất tại mỗi mốc thời gian.
- `Low`: giá thấp nhất tại mỗi mốc thời gian.
- `Close`: giá đóng cửa.
- `Volume`: khối lượng giao dịch.

Mục tiêu của bài không chỉ là huấn luyện được mô hình RNN, mà còn phải đánh giá xem mô hình có dự đoán tốt hơn một phương pháp cơ sở đơn giản hay không. Phương pháp cơ sở được dùng là:

```text
Giá dự đoán ngày mai = Giá đóng cửa hôm nay
```

Baseline này phù hợp với chuỗi giá tài chính vì giá ngày mai thường rất gần giá hôm nay. Nếu mô hình học máy không tốt hơn baseline này thì mô hình chưa có nhiều giá trị dự đoán thực tế.

## 2. Dữ liệu và tiền xử lý

Dữ liệu gốc nằm trong file:

```text
archive.zip
```

Bên trong file zip có file CSV:

```text
btcusd_1-min_data.csv
```

Đây là dữ liệu Bitcoin theo từng phút, kích thước rất lớn. Nếu đưa trực tiếp toàn bộ dữ liệu từng phút vào RNN thì mô hình sẽ rất nặng, thời gian huấn luyện lâu và dễ bị nhiễu. Vì vậy bài làm chuyển dữ liệu về tần suất ngày.

Quá trình tiền xử lý gồm các bước:

1. Đọc file CSV từ `archive.zip`.
2. Chuyển `Timestamp` thành kiểu `datetime`.
3. Sắp xếp dữ liệu theo thời gian tăng dần.
4. Đặt cột thời gian làm index.
5. Resample dữ liệu từ từng phút thành dữ liệu theo ngày.

Cách tổng hợp OHLCV theo ngày:

```text
Open   = giá đầu tiên trong ngày
High   = giá cao nhất trong ngày
Low    = giá thấp nhất trong ngày
Close  = giá cuối cùng trong ngày
Volume = tổng volume trong ngày
```

Sau khi resample, các ngày thiếu dữ liệu OHLC sẽ bị loại bỏ. Cách xử lý này phù hợp với dữ liệu chuỗi thời gian vì mỗi dòng dữ liệu sau cùng đại diện cho một ngày giao dịch.

## 3. Feature engineering

Dữ liệu gốc chỉ có OHLCV. Nếu chỉ đưa các cột này vào RNN thì mô hình có thể học được một phần biến động giá, nhưng chưa có nhiều thông tin về xu hướng, độ biến động và volume. Vì vậy bài làm tạo thêm các đặc trưng kỹ thuật từ dữ liệu giá và volume.

Nhóm đặc trưng giá gốc:

```text
Open, High, Low, Close, Volume
```

Nhóm đặc trưng return:

```text
return_1d
log_return_1d
return_lag_2
return_lag_3
return_lag_7
return_lag_14
abs_return_1d
```

Ý nghĩa:

- `return_1d`: tỷ lệ thay đổi giá đóng cửa so với ngày trước.
- `log_return_1d`: log return, thường được dùng trong tài chính vì ổn định hơn return thông thường.
- `return_lag_*`: return của các ngày trước đó, giúp mô hình có thêm thông tin về động lượng giá trong quá khứ.
- `abs_return_1d`: độ lớn của biến động giá, không quan tâm giá tăng hay giảm.

Nhóm đặc trưng biến động trong ngày:

```text
price_range
range_pct
body_pct
typical_price
```

Ý nghĩa:

- `price_range = High - Low`: biên độ giá trong ngày.
- `range_pct = price_range / Close`: biên độ giá tính theo tỷ lệ.
- `body_pct = (Close - Open) / Open`: thân nến của ngày, cho biết giá đóng cửa cao hay thấp hơn giá mở cửa.
- `typical_price = (High + Low + Close) / 3`: mức giá đại diện trung bình trong ngày.

Nhóm moving average và volatility:

```text
ma_7, ma_14, ma_30
std_7, std_14, std_30
close_to_ma_7, close_to_ma_14, close_to_ma_30
```

Ý nghĩa:

- `ma_*`: đường trung bình động, giúp mô hình nhận biết xu hướng ngắn hạn và trung hạn.
- `std_*`: độ lệch chuẩn rolling, đại diện cho volatility.
- `close_to_ma_*`: khoảng cách tương đối từ giá hiện tại đến moving average, giúp mô hình biết giá đang cao hay thấp so với xu hướng gần đây.

Nhóm đặc trưng volume:

```text
volume_ma_7, volume_ma_14, volume_ma_30
volume_ratio_7, volume_ratio_14, volume_ratio_30
log_volume
```

Ý nghĩa:

- `volume_ma_*`: volume trung bình động.
- `volume_ratio_*`: volume hiện tại so với volume trung bình, giúp phát hiện những ngày có giao dịch bất thường.
- `log_volume`: biến đổi log để giảm độ lệch của volume vì volume tài chính thường phân phối lệch.

Nhóm chỉ báo kỹ thuật:

```text
rsi_14
bb_position_20
macd
macd_signal
macd_hist
```

Ý nghĩa:

- `rsi_14`: Relative Strength Index, đo sức mạnh tăng hoặc giảm trong 14 ngày.
- `bb_position_20`: vị trí của giá trong Bollinger Bands 20 ngày.
- `macd`: chênh lệch giữa EMA 12 và EMA 26.
- `macd_signal`: đường tín hiệu EMA 9 của MACD.
- `macd_hist`: độ chênh giữa MACD và đường signal.

Các giá trị vô cùng hoặc thiếu dữ liệu sau khi tính rolling window được thay bằng `NaN` và loại bỏ bằng `dropna()`.

## 4. Target dự đoán

Phiên bản ban đầu dự đoán trực tiếp:

```text
target_next_close = Close ngày mai
```

Cách này dễ làm mô hình bị lệch khi giá Bitcoin trong tập test nằm ở vùng giá khác với tập train. Giá BTC thay đổi mạnh theo thời gian, nên dự đoán giá tuyệt đối thường khó hơn.

Phiên bản hiện tại chuyển sang dự đoán return:

```text
target_next_return = target_next_close / Close hôm nay - 1
```

Sau khi mô hình dự đoán return, giá dự đoán được quy đổi lại:

```text
predicted_close = Close hôm nay * (1 + predicted_return)
```

Cách này hợp lý hơn cho dữ liệu tài chính vì mô hình học tỷ lệ thay đổi giá, thay vì học trực tiếp mức giá tuyệt đối.

## 5. Chia dữ liệu train, validation và test

Dữ liệu time series không được trộn lẫn. Nếu trộn lẫn, mô hình có thể nhìn thấy thông tin từ tương lai trong quá trình huấn luyện, gây rò rỉ dữ liệu.

Bài làm chia dữ liệu theo thứ tự thời gian:

```text
Train      = 70% đầu tiên
Validation = 15% tiếp theo
Test       = 15% cuối cùng
```

Ý nghĩa:

- Train set dùng để fit scaler và huấn luyện mô hình.
- Validation set dùng để theo dõi `val_loss`, chọn best model và early stopping.
- Test set chỉ dùng để đánh giá cuối cùng sau khi mô hình đã huấn luyện xong.

Quan trọng: scaler chỉ được fit trên train set. Validation và test chỉ được transform bằng scaler đã học từ train. Cách này tránh rò rỉ thông tin từ validation hoặc test vào quá trình huấn luyện.

## 6. Tạo sequence cho RNN

RNN không nhận từng dòng dữ liệu độc lập như Linear Regression. RNN nhận chuỗi dữ liệu có dạng 3 chiều:

```text
(samples, timesteps, features)
```

Trong bài này:

```text
sequence_length = 30
```

Nghĩa là mỗi sample gồm 30 ngày gần nhất để dự đoán return của ngày tiếp theo.

Ví dụ:

```text
Input  = dữ liệu từ ngày t-29 đến ngày t
Target = return của ngày t+1
```

Hàm `make_sequences()` tạo các cửa sổ trượt 30 ngày từ dữ liệu đã scale. Validation và test được ghép thêm 30 ngày cuối của split trước làm context, nhưng scaler vẫn chỉ fit trên train. Điều này giúp tạo được sequence đầu tiên của validation/test mà không làm rò rỉ target tương lai.

## 7. Mô hình RNN

Mô hình được xây bằng TensorFlow/Keras. Kiến trúc tổng quát:

```text
Input shape = (sequence_length, number_of_features)
RNN layer   = LSTM hoặc GRU
Dropout
Dense 32 ReLU
Dense 1
```

Cụ thể:

```text
Input: 30 ngày x số feature
RNN: LSTM/GRU 64 units
Dropout: 0.2
Dense: 32 neurons, activation ReLU
Output: 1 giá trị predicted return
```

Model hỗ trợ hai loại recurrent layer:

```text
--rnn-type lstm
--rnn-type gru
```

Mặc định đang dùng `lstm`.

Model cũng hỗ trợ Bidirectional RNN. Mặc định bidirectional được bật. Nếu muốn tắt thì dùng:

```bash
python train.py --no-bidirectional
```

Bidirectional RNN đọc chuỗi input theo hai hướng trong cùng cửa sổ 30 ngày. Với bài này, nó không nhìn vào tương lai sau ngày cần dự đoán, vì input chỉ gồm 30 ngày quá khứ. Do đó cách dùng này không gây rò rỉ target.

## 8. Kỹ thuật huấn luyện

Optimizer:

```text
Adam
```

Loss:

```text
MSE
```

Metric trong quá trình train:

```text
MAE
```

Vì target là `target_next_return`, loss và MAE trong log train là lỗi trên return đã scale, không phải lỗi giá USD trực tiếp.

Callbacks được sử dụng:

```text
ModelCheckpoint
EarlyStopping
ReduceLROnPlateau
```

Ý nghĩa:

- `ModelCheckpoint`: lưu model có `val_loss` tốt nhất vào `artifacts/best_model.keras`.
- `EarlyStopping`: dừng train nếu validation loss không cải thiện trong nhiều epoch, đồng thời khôi phục weights tốt nhất.
- `ReduceLROnPlateau`: giảm learning rate khi validation loss bị chững lại, giúp mô hình tối ưu mịn hơn.

Trong lần train này đặt:

```text
epochs = 100
batch_size = 64
```

Tuy nhiên mô hình dừng ở epoch 12 do EarlyStopping. Đây là bình thường, vì `epochs=100` là số epoch tối đa, không phải bắt buộc train đủ 100 epoch.

## 9. K-Fold cho dữ liệu time series

Bài làm có áp dụng cross-validation cho time series bằng:

```text
TimeSeriesSplit
```

Mặc định:

```text
cv_folds = 5
```

Khác với KFold thông thường, `TimeSeriesSplit` không shuffle dữ liệu. Mỗi fold dùng đoạn quá khứ để train và đoạn sau đó để validation. Cách này phù hợp với dữ liệu time series vì trong thực tế ta chỉ có thể dùng quá khứ để dự đoán tương lai.

Nếu chạy:

```bash
python train.py
```

thì sẽ chạy TimeSeriesSplit CV trước, sau đó train final model.



## 10. Kết quả huấn luyện

Kết quả từ lần train trên Google Colab GPU T4:

```text
Saved feature CSV: bitcoin_features_daily.csv
Saved best model: artifacts/best_model.keras
Saved final model: artifacts/final_model.keras

Validation loss: 0.001050
Validation return MAE scaled: 0.022527

Test loss: 0.000851
Test return MAE scaled: 0.021252

Validation price scores:
MAE  = 626.1815
RMSE = 955.8410
MAPE = 2.02%

Test price scores:
MAE  = 1598.1593
RMSE = 2192.8090
MAPE = 1.91%
```

Giải thích:

- `Validation loss` và `Test loss` là MSE trên target return đã scale.
- `Validation return MAE scaled` và `Test return MAE scaled` là MAE trên return đã scale.
- `Validation price scores` và `Test price scores` là chỉ số sau khi quy đổi predicted return về predicted close.

Chỉ số quan trọng để đọc kết quả cuối cùng là:

```text
Test MAPE = 1.91%
```

Nghĩa là trên test set, dự đoán của mô hình lệch trung bình khoảng 1.91% so với giá BTC thật của ngày tiếp theo.

## 11. Kết quả test.py và so sánh baseline

Sau khi train, chạy:

```bash
python test.py
```

Kết quả:

```text
Model file: artifacts/best_model.keras

model                 MAE          RMSE        MAPE_percent    direction_accuracy_percent
rnn                   1598.159235  2192.809221 1.912837        48.531290
naive_previous_close  1453.577806  2011.647236 1.750730        49.425287
```

Nhận xét:

- Model RNN có `MAPE = 1.91%`.


Điều này cho thấy bài toán dự đoán giá Bitcoin ngày tiếp theo rất khó. Giá ngày mai thường rất gần giá hôm nay, nên baseline `previous close` là một baseline rất mạnh.

Tuy nhiên, kết quả hiện tại vẫn có ý nghĩa:

- MAPE đã giảm về mức gần baseline.
- Validation MAPE 2.02% và Test MAPE 1.91% khá gần nhau, không có dấu hiệu overfitting quá nặng.
- Việc chuyển target sang return giúp mô hình ổn định hơn và giảm lỗi đáng kể.

## 12. Direction accuracy

`direction_accuracy_percent` đo tỷ lệ mô hình dự đoán đúng hướng thay đổi giá. Kết quả:

```text
RNN direction accuracy      = 48.53%
Naive direction accuracy    = 49.43%
```

Chỉ số này thấp hơn 50%, nghĩa là mô hình chưa dự đoán hướng tăng/giảm tốt. Điều này thường gặp với dữ liệu tài chính ngắn hạn, vì biến động ngày của Bitcoin rất nhiễu và khó dự đoán.

Nếu mục tiêu của bài toán là trading, direction accuracy rất quan trọng. Nếu mục tiêu của bài là dự đoán giá và thực hành RNN, thì MAPE/MAE/RMSE là các chỉ số chính cần báo cáo.



## 13. Kết luận

Bài làm đã xây dựng pipeline dự đoán giá Bitcoin bằng RNN từ đầu đến cuối:

1. Đọc dữ liệu Bitcoin theo từng phút.
2. Resample thành dữ liệu ngày.
3. Tạo feature kỹ thuật từ OHLCV.
4. Chuyển target từ giá tuyệt đối sang return ngày tiếp theo.
5. Scale dữ liệu đúng cách, chỉ fit scaler trên train.
6. Tạo sequence 30 ngày cho RNN.
7. Train model LSTM/GRU có hỗ trợ Bidirectional.
8. Áp dụng TimeSeriesSplit cross-validation cho time series.
9. Dùng checkpoint, early stopping và reduce learning rate.
10. Đánh giá bằng MAE, RMSE, MAPE và direction accuracy.
11. So sánh với baseline naive previous close.

Kết quả cuối cùng:

```text
RNN Test MAPE      = 1.91%
Baseline Test MAPE = 1.75%
```

Model RNN đã đạt kết quả gần baseline và tốt hơn nhiều so với cách dự đoán giá tuyệt đối. Tuy nhiên, model vẫn chưa vượt baseline naive. Vì vậy, kết luận phù hợp là: RNN có thể học được một phần pattern từ dữ liệu Bitcoin, nhưng dự đoán giá BTC ngày tiếp theo vẫn rất khó và baseline previous-close vẫn là một đối thủ rất mạnh.

Hướng cải thiện tiếp theo:

- Thử `sequence_length` 14, 60 hoặc 90 ngày.
- So sánh LSTM, GRU, Bidirectional LSTM và Bidirectional GRU.
- Tối ưu hyperparameter: units, dropout, learning rate, batch size.
- Thử dự đoán multi-step hoặc direction classification.
- Thêm feature từ thị trường khác như SP500, DXY, gold, interest rate hoặc sentiment nếu có dữ liệu.
