import argparse
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error


def add_features(daily: pd.DataFrame, target_column: str) -> pd.DataFrame:
    features = daily.copy()
    features["return_1d"] = features["Close"].pct_change()
    features["log_return_1d"] = np.log(features["Close"]).diff()
    for lag in [2, 3, 7, 14]:
        features[f"return_lag_{lag}"] = features["return_1d"].shift(lag)

    features["price_range"] = features["High"] - features["Low"]
    features["range_pct"] = features["price_range"] / features["Close"]
    features["body"] = features["Close"] - features["Open"]
    features["body_pct"] = features["body"] / features["Open"]
    features["typical_price"] = (
        features["High"] + features["Low"] + features["Close"]
    ) / 3
    features["log_volume"] = np.log1p(features["Volume"])
    features["abs_return_1d"] = features["return_1d"].abs()

    for window in [7, 14, 30]:
        features[f"ma_{window}"] = features["Close"].rolling(window).mean()
        features[f"std_{window}"] = features["Close"].rolling(window).std()
        features[f"volume_ma_{window}"] = features["Volume"].rolling(window).mean()
        features[f"close_to_ma_{window}"] = (
            features["Close"] / features[f"ma_{window}"] - 1
        )
        features[f"volume_ratio_{window}"] = (
            features["Volume"] / features[f"volume_ma_{window}"]
        )

    delta = features["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    features["rsi_14"] = 100 - (100 / (1 + rs))

    ma_20 = features["Close"].rolling(20).mean()
    std_20 = features["Close"].rolling(20).std()
    lower_band = ma_20 - 2 * std_20
    upper_band = ma_20 + 2 * std_20
    features["bb_position_20"] = (
        features["Close"] - lower_band
    ) / (upper_band - lower_band)

    ema_12 = features["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = features["Close"].ewm(span=26, adjust=False).mean()
    features["macd"] = ema_12 - ema_26
    features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
    features["macd_hist"] = features["macd"] - features["macd_signal"]

    features["target_next_close"] = features["Close"].shift(-1)
    features["target_next_return"] = features["target_next_close"] / features["Close"] - 1
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.dropna().copy()


def build_feature_csv(
    archive_path: Path,
    output_path: Path,
    target_column: str,
) -> pd.DataFrame:
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Could not find {archive_path}. Upload archive.zip or provide --data-csv."
        )

    dtype_map = {
        "Timestamp": "float64",
        "Open": "float32",
        "High": "float32",
        "Low": "float32",
        "Close": "float32",
        "Volume": "float32",
    }

    with zipfile.ZipFile(archive_path) as zf:
        csv_files = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_files:
            raise ValueError(f"No CSV file found inside {archive_path}")
        with zf.open(csv_files[0]) as file:
            raw = pd.read_csv(file, dtype=dtype_map)

    raw["datetime"] = pd.to_datetime(raw["Timestamp"], unit="s", utc=True)
    raw = raw.sort_values("datetime").set_index("datetime")

    daily = raw.resample("1D").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    daily = daily.dropna(subset=["Open", "High", "Low", "Close"])

    features = add_features(daily, target_column)
    features.index.name = "datetime"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path)
    return features


def load_feature_data(
    data_csv: Path,
    archive_path: Path,
    target_column: str,
    required_columns: list[str],
) -> pd.DataFrame:
    if data_csv.exists():
        data = pd.read_csv(data_csv, parse_dates=["datetime"], index_col="datetime")
        data = data.sort_index()
        if set(required_columns).issubset(data.columns):
            return data
        print(f"Rebuilding {data_csv} because it is missing new feature columns.")
    return build_feature_csv(
        archive_path=archive_path,
        output_path=data_csv,
        target_column=target_column,
    )


def split_dataframe(data: pd.DataFrame, train_ratio: float, val_ratio: float):
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    train_df = data.iloc[:train_size].copy()
    val_df = data.iloc[train_size : train_size + val_size].copy()
    test_df = data.iloc[train_size + val_size :].copy()
    return train_df, val_df, test_df


def make_sequences(x: np.ndarray, y: np.ndarray, sequence_length: int):
    xs, ys = [], []
    for i in range(sequence_length, len(x)):
        xs.append(x[i - sequence_length : i])
        ys.append(y[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    direction_true = np.sign(np.diff(y_true))
    direction_pred = np.sign(np.diff(y_pred))
    direction_accuracy = float(np.mean(direction_true == direction_pred) * 100)
    return {
        "MAE": float(mae),
        "RMSE": rmse,
        "MAPE_percent": mape,
        "direction_accuracy_percent": direction_accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained RNN model.")
    parser.add_argument("--data-csv", type=Path, default=Path("bitcoin_features_daily.csv"))
    parser.add_argument("--archive", type=Path, default=Path("archive.zip"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--model-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("scores"))
    args = parser.parse_args()

    metadata_path = args.model_dir / "metadata.json"
    scalers_path = args.model_dir / "scalers.pkl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}. Run train.py first.")
    if not scalers_path.exists():
        raise FileNotFoundError(f"Missing {scalers_path}. Run train.py first.")

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata["feature_columns"]
    target_column = metadata["target_column"]
    price_target_column = metadata.get("price_target_column", "target_next_close")
    sequence_length = int(metadata["sequence_length"])
    train_ratio = float(metadata["train_ratio"])
    val_ratio = float(metadata["val_ratio"])

    required_columns = feature_columns + [target_column, price_target_column]
    data = load_feature_data(
        args.data_csv,
        args.archive,
        target_column,
        required_columns,
    )
    missing_columns = [col for col in required_columns if col not in data]
    if missing_columns:
        raise ValueError(f"Missing columns in data: {missing_columns}")

    _, val_df, test_df = split_dataframe(data, train_ratio, val_ratio)
    if len(test_df) <= sequence_length:
        raise ValueError("Test split must contain more rows than sequence_length.")

    with open(scalers_path, "rb") as file:
        scalers = pickle.load(file)
    x_scaler = scalers["x_scaler"]
    y_scaler = scalers["y_scaler"]

    test_context = pd.concat([val_df.tail(sequence_length), test_df])
    x_test = x_scaler.transform(test_context[feature_columns])
    y_test = y_scaler.transform(test_context[[target_column]])
    X_test, Y_test = make_sequences(x_test, y_test, sequence_length)

    model_path = args.model_file
    if model_path is None:
        best_model = args.model_dir / "best_model.keras"
        final_model = args.model_dir / "final_model.keras"
        model_path = best_model if best_model.exists() else final_model
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    model = tf.keras.models.load_model(model_path)
    pred_scaled = model.predict(X_test, verbose=0)

    test_index = test_df.index
    current_close = test_df["Close"].to_numpy()
    predicted_target = y_scaler.inverse_transform(pred_scaled).reshape(-1)

    if target_column == "target_next_return":
        y_true_return = y_scaler.inverse_transform(Y_test).reshape(-1)
        y_pred_return = predicted_target
        y_true = test_df[price_target_column].to_numpy()
        y_pred = current_close * (1 + y_pred_return)
    else:
        y_true = y_scaler.inverse_transform(Y_test).reshape(-1)
        y_pred = predicted_target
        y_true_return = test_df["target_next_return"].to_numpy()
        y_pred_return = y_pred / current_close - 1

    model_scores = regression_scores(y_true, y_pred)
    naive_pred = current_close
    baseline_scores = regression_scores(y_true, naive_pred)

    predictions = pd.DataFrame(
        {
            "datetime": test_index,
            "actual": y_true,
            "predicted": y_pred,
            "naive_predicted": naive_pred,
            "actual_return": y_true_return,
            "predicted_return": y_pred_return,
            "error": y_true - y_pred,
            "absolute_error": np.abs(y_true - y_pred),
        }
    )

    scores = pd.DataFrame(
        [
            {"model": "rnn", **model_scores},
            {"model": "naive_previous_close", **baseline_scores},
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.csv"
    scores_csv_path = args.output_dir / "scores.csv"
    scores_txt_path = args.output_dir / "scores.txt"

    predictions.to_csv(predictions_path, index=False)
    scores.to_csv(scores_csv_path, index=False)

    with open(scores_txt_path, "w", encoding="utf-8") as file:
        file.write(f"Model file: {model_path}\n")
        file.write(f"Test rows after sequence cut: {len(predictions)}\n\n")
        file.write(scores.to_string(index=False))
        file.write("\n")

    print(f"Model file: {model_path}")
    print(scores.to_string(index=False))
    print(f"Saved scores CSV: {scores_csv_path}")
    print(f"Saved scores TXT: {scores_txt_path}")
    print(f"Saved predictions: {predictions_path}")


if __name__ == "__main__":
    main()
