import argparse
import json
import pickle
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, GRU, LSTM
from tensorflow.keras.models import Sequential


FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "return_1d",
    "log_return_1d",
    "return_lag_2",
    "return_lag_3",
    "return_lag_7",
    "return_lag_14",
    "price_range",
    "range_pct",
    "body_pct",
    "typical_price",
    "log_volume",
    "abs_return_1d",
    "ma_7",
    "ma_14",
    "ma_30",
    "std_7",
    "std_14",
    "std_30",
    "volume_ma_7",
    "volume_ma_14",
    "volume_ma_30",
    "close_to_ma_7",
    "close_to_ma_14",
    "close_to_ma_30",
    "volume_ratio_7",
    "volume_ratio_14",
    "volume_ratio_30",
    "rsi_14",
    "bb_position_20",
    "macd",
    "macd_signal",
    "macd_hist",
]
TARGET_COLUMN = "target_next_return"
PRICE_TARGET_COLUMN = "target_next_close"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def add_features(daily: pd.DataFrame) -> pd.DataFrame:
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

    features[PRICE_TARGET_COLUMN] = features["Close"].shift(-1)
    features[TARGET_COLUMN] = features[PRICE_TARGET_COLUMN] / features["Close"] - 1
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.dropna().copy()


def build_feature_csv(archive_path: Path, output_path: Path) -> pd.DataFrame:
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

    features = add_features(daily)
    features.index.name = "datetime"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path)
    return features


def load_feature_data(data_csv: Path, archive_path: Path) -> pd.DataFrame:
    if data_csv.exists():
        data = pd.read_csv(data_csv, parse_dates=["datetime"], index_col="datetime")
        data = data.sort_index()
        required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN, PRICE_TARGET_COLUMN])
        if required_columns.issubset(data.columns):
            return data
        print(f"Rebuilding {data_csv} because it is missing new feature columns.")
    return build_feature_csv(archive_path=archive_path, output_path=data_csv)


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


def prepare_split_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    sequence_length: int,
):
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_train = x_scaler.fit_transform(train_df[FEATURE_COLUMNS])
    y_train = y_scaler.fit_transform(train_df[[TARGET_COLUMN]])

    # Validation/test sequences should be allowed to use the last training
    # window as context, but scalers are still fitted only on training data.
    val_context = pd.concat([train_df.tail(sequence_length), val_df])
    x_val = x_scaler.transform(val_context[FEATURE_COLUMNS])
    y_val = y_scaler.transform(val_context[[TARGET_COLUMN]])

    X_train, Y_train = make_sequences(x_train, y_train, sequence_length)
    X_val, Y_val = make_sequences(x_val, y_val, sequence_length)
    return X_train, Y_train, X_val, Y_val, x_scaler, y_scaler


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"mae": float(mae), "rmse": rmse, "mape_percent": mape}


def build_model(
    sequence_length: int,
    n_features: int,
    units: int,
    dropout: float,
    learning_rate: float,
    bidirectional: bool,
    rnn_type: str,
) -> tf.keras.Model:
    rnn_layer_cls = LSTM if rnn_type == "lstm" else GRU
    rnn_layer = rnn_layer_cls(units, return_sequences=False)
    if bidirectional:
        rnn_layer = Bidirectional(rnn_layer)

    model = Sequential(
        [
            tf.keras.Input(shape=(sequence_length, n_features)),
            rnn_layer,
            Dropout(dropout),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_callbacks(model_path: Path):
    return [
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]


def run_time_series_cv(
    data: pd.DataFrame,
    args: argparse.Namespace,
    bidirectional: bool,
) -> pd.DataFrame:
    if args.cv_folds < 2:
        return pd.DataFrame()

    splitter = TimeSeriesSplit(n_splits=args.cv_folds)
    cv_rows = []
    cv_dir = args.model_dir / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(data), start=1):
        fold_train_df = data.iloc[train_idx].copy()
        fold_val_df = data.iloc[val_idx].copy()
        if min(len(fold_train_df), len(fold_val_df)) <= args.sequence_length:
            raise ValueError(
                f"Fold {fold} is too small for sequence_length={args.sequence_length}."
            )

        X_train, Y_train, X_val, Y_val, _, y_scaler = prepare_split_data(
            train_df=fold_train_df,
            val_df=fold_val_df,
            sequence_length=args.sequence_length,
        )

        tf.keras.backend.clear_session()
        model = build_model(
            sequence_length=args.sequence_length,
            n_features=len(FEATURE_COLUMNS),
            units=args.units,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            bidirectional=bidirectional,
            rnn_type=args.rnn_type,
        )

        fold_model_path = cv_dir / f"fold_{fold}_best_model.keras"
        print(
            f"\nCV fold {fold}/{args.cv_folds}: "
            f"train {fold_train_df.index.min()} -> {fold_train_df.index.max()} | "
            f"val {fold_val_df.index.min()} -> {fold_val_df.index.max()}"
        )

        history = model.fit(
            X_train,
            Y_train,
            validation_data=(X_val, Y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=build_callbacks(fold_model_path),
            shuffle=False,
            verbose=args.verbose,
        )

        pred_scaled = model.predict(X_val, verbose=0)
        y_pred_return = y_scaler.inverse_transform(pred_scaled).reshape(-1)
        y_true_price = fold_val_df[PRICE_TARGET_COLUMN].to_numpy()
        y_pred_price = fold_val_df["Close"].to_numpy() * (1 + y_pred_return)
        scores = regression_scores(y_true_price, y_pred_price)
        best_val_loss = float(np.min(history.history["val_loss"]))
        best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

        cv_rows.append(
            {
                "fold": fold,
                "train_start": str(fold_train_df.index.min()),
                "train_end": str(fold_train_df.index.max()),
                "val_start": str(fold_val_df.index.min()),
                "val_end": str(fold_val_df.index.max()),
                "train_rows": len(fold_train_df),
                "val_rows": len(fold_val_df),
                "X_train_samples": len(X_train),
                "X_val_samples": len(X_val),
                "best_epoch": best_epoch,
                "best_val_loss_scaled": best_val_loss,
                **scores,
            }
        )

    cv_results = pd.DataFrame(cv_rows)
    cv_results.to_csv(args.model_dir / "cv_results.csv", index=False)

    summary = cv_results[["mae", "rmse", "mape_percent", "best_val_loss_scaled"]].agg(
        ["mean", "std"]
    )
    summary.to_csv(args.model_dir / "cv_summary.csv")
    print("\nTimeSeriesSplit CV results:")
    print(cv_results.to_string(index=False))
    print("\nCV summary:")
    print(summary.to_string())
    return cv_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RNN model for Bitcoin price.")
    parser.add_argument("--data-csv", type=Path, default=Path("bitcoin_features_daily.csv"))
    parser.add_argument("--archive", type=Path, default=Path("archive.zip"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--rnn-type", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--no-bidirectional", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    data = load_feature_data(args.data_csv, args.archive)
    missing_columns = [
        col
        for col in FEATURE_COLUMNS + [TARGET_COLUMN, PRICE_TARGET_COLUMN]
        if col not in data
    ]
    if missing_columns:
        raise ValueError(f"Missing columns in data: {missing_columns}")

    train_df, val_df, test_df = split_dataframe(data, args.train_ratio, args.val_ratio)
    if min(len(train_df), len(val_df), len(test_df)) <= args.sequence_length:
        raise ValueError("Each split must contain more rows than --sequence-length.")

    bidirectional = not args.no_bidirectional
    train_val_df = pd.concat([train_df, val_df])

    cv_results = pd.DataFrame()
    if not args.skip_cv and args.cv_folds >= 2:
        cv_results = run_time_series_cv(
            data=train_val_df,
            args=args,
            bidirectional=bidirectional,
        )

    X_train, Y_train, X_val, Y_val, x_scaler, y_scaler = prepare_split_data(
        train_df=train_df,
        val_df=val_df,
        sequence_length=args.sequence_length,
    )

    test_context = pd.concat([val_df.tail(args.sequence_length), test_df])
    x_test = x_scaler.transform(test_context[FEATURE_COLUMNS])
    y_test = y_scaler.transform(test_context[[TARGET_COLUMN]])
    X_test, Y_test = make_sequences(x_test, y_test, args.sequence_length)

    tf.keras.backend.clear_session()
    model = build_model(
        sequence_length=args.sequence_length,
        n_features=len(FEATURE_COLUMNS),
        units=args.units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        bidirectional=bidirectional,
        rnn_type=args.rnn_type,
    )
    model.summary()

    best_model_path = args.model_dir / "best_model.keras"

    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=build_callbacks(best_model_path),
        shuffle=False,
        verbose=args.verbose,
    )

    final_model_path = args.model_dir / "final_model.keras"
    model.save(final_model_path)

    with open(args.model_dir / "scalers.pkl", "wb") as file:
        pickle.dump({"x_scaler": x_scaler, "y_scaler": y_scaler}, file)

    metadata = {
        "data_csv": str(args.data_csv),
        "archive": str(args.archive),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "price_target_column": PRICE_TARGET_COLUMN,
        "sequence_length": args.sequence_length,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "cv_folds": 0 if args.skip_cv else args.cv_folds,
        "rnn_type": args.rnn_type,
        "bidirectional": bidirectional,
        "units": args.units,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "X_train_shape": list(X_train.shape),
        "X_val_shape": list(X_val.shape),
        "X_test_shape": list(X_test.shape),
        "train_start": str(train_df.index.min()),
        "train_end": str(train_df.index.max()),
        "val_start": str(val_df.index.min()),
        "val_end": str(val_df.index.max()),
        "test_start": str(test_df.index.min()),
        "test_end": str(test_df.index.max()),
    }
    if not cv_results.empty:
        metadata["cv_results_csv"] = str(args.model_dir / "cv_results.csv")
        metadata["cv_summary_csv"] = str(args.model_dir / "cv_summary.csv")
    with open(args.model_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    pd.DataFrame(history.history).to_csv(args.model_dir / "history.csv", index=False)

    val_loss, val_mae = model.evaluate(X_val, Y_val, verbose=0)
    test_loss, test_mae = model.evaluate(X_test, Y_test, verbose=0)
    val_pred_return = y_scaler.inverse_transform(model.predict(X_val, verbose=0)).reshape(
        -1
    )
    test_pred_return = y_scaler.inverse_transform(
        model.predict(X_test, verbose=0)
    ).reshape(-1)
    val_price_scores = regression_scores(
        val_df[PRICE_TARGET_COLUMN].to_numpy(),
        val_df["Close"].to_numpy() * (1 + val_pred_return),
    )
    test_price_scores = regression_scores(
        test_df[PRICE_TARGET_COLUMN].to_numpy(),
        test_df["Close"].to_numpy() * (1 + test_pred_return),
    )
    print(f"Saved feature CSV: {args.data_csv}")
    print(f"Saved best model: {best_model_path}")
    print(f"Saved final model: {final_model_path}")
    print(f"Validation loss: {val_loss:.6f} | Validation return MAE scaled: {val_mae:.6f}")
    print(f"Test loss: {test_loss:.6f} | Test return MAE scaled: {test_mae:.6f}")
    print(
        "Validation price scores: "
        f"MAE={val_price_scores['mae']:.4f} | "
        f"RMSE={val_price_scores['rmse']:.4f} | "
        f"MAPE={val_price_scores['mape_percent']:.2f}%"
    )
    print(
        "Test price scores: "
        f"MAE={test_price_scores['mae']:.4f} | "
        f"RMSE={test_price_scores['rmse']:.4f} | "
        f"MAPE={test_price_scores['mape_percent']:.2f}%"
    )


if __name__ == "__main__":
    main()
