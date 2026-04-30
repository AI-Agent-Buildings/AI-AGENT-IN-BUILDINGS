

import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# ── CONFIG ──────────────────────────────────────────────────────
CSV_PATH   = r"C:\Users\hannan\Documents\Uni\sensor-data.csv"
SEQ_LEN    = 7        # days of history the LSTM sees( used as input )
UNITS      = 20       # LSTM memory cells (kept small so weights fit in n8n)
EPOCHS     = 40       # sweet spot — more than this and we start memorising noise
BATCH      = 8        # small batch size improves learning stability for time-series data like this
ROOM_IDS   = [1, 2, 3]
ROOM_NAMES = {1:"Kitchen", 2:"Data Lab", 3:"Mailroom"}
OUT_FILE   = "lstm_weights.json" #Outputs the trained weight in json files 




def load_data():
    df = pd.read_csv(CSV_PATH) #Reads CSV file
    required = {"timestamp","temperature","humidity","room_id"} #Ensures dataset has required columns
    missing  = required - set(df.columns) ## find any that are absent
    if missing: # if anything is missing, stop now
        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")
        
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True) # parse timestamps, force UTC to avoid timezone bugs
    df = df.sort_values("timestamp").dropna(subset=["temperature","humidity"]) # sort oldest-first, drop rows with no sensor reading
    print(f"  Loaded {len(df)} rows  {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Rooms in dataset: {sorted(df['room_id'].unique())}")
    return df # hand the clean dataframe back to the caller


def daily_resample(df_room):
  
    return (
        df_room.set_index("timestamp")[["temperature","humidity"]] # make timestamp the index, keep only the two sensor columns
        .resample("1D").mean()       # group all readings in each calendar day and average them
        .interpolate("linear")       # fill any day with no readings by drawing a straight line between neighbours
        .dropna()                    # remove any remaining NaNs that interpolation couldn't fix (e.g. at the edges)
        .reset_index()               # turn the timestamp index back into a normal column
    )


def make_seqs(arr, seq_len):
    X, y = [], []  # X = input windows, y = the next value after each window
    for i in range(len(arr) - seq_len): # stop early so we always have a label to pair with the window
        X.append(arr[i:i+seq_len])# grab seq_len consecutive rows as one input sample
        y.append(arr[i+seq_len]) # the row immediately after the window is the target
    return np.array(X, dtype="float32"), np.array(y, dtype="float32") # convert lists to float32 arrays that Keras expects


def r2(true, pred):
    # sum of squared differences between predictions and actuals
    # total variance in the actual values
    # R² formula; returns 0 if actuals are all the same (flat line edge case)
    ss_res = np.sum((true - pred)**2)
    ss_tot = np.sum((true - np.mean(true))**2)
    return 1 - ss_res/ss_tot if ss_tot > 0 else 0.0


def train_room(room_id, df_room):
    #Data Preperation Filters and resamples the room's data, checks there's enough to work with,
    #scales temperature and humidity to [0, 1] using MinMaxScaler
    #then builds the sliding-window sequences and splits them 80/20 into train and test sets.
    name = ROOM_NAMES.get(room_id, f"Room {room_id}")
    print(f"\n{'─'*55}")
    print(f"  Training Room {room_id} — {name}")
    print(f"{'─'*55}")

    daily = daily_resample(df_room)
    print(f"  Raw readings: {len(df_room)}  →  Daily points: {len(daily)}")

    if len(daily) < SEQ_LEN * 3:
        raise ValueError(
            f"Room {room_id}: only {len(daily)} daily rows. "
            f"Need ≥ {SEQ_LEN*3}. Reduce SEQ_LEN or add more data."
        )

    vals   = daily[["temperature","humidity"]].values.astype("float32")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(vals)

    X, y = make_seqs(scaled, SEQ_LEN)
    split    = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]


        # build the model layer by layer
        # tell Keras: each sample is SEQ_LEN timesteps × 2 features
        # LSTM reads the sequence and compresses it into a hidden state vector
        #map that hidden state to two outputs: tomorrow's temperature and humidity
 
    model = Sequential([
        Input(shape=(SEQ_LEN, 2)),
        LSTM(UNITS, activation="tanh"),
        Dense(2),
    ])
    model.compile(optimizer="adam", loss="mse") #  adam optimizer handles learning rate automatically; mse penalises large errors more
    model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH,
              validation_data=(X_te, y_te), verbose=1) # train and watch val loss each epoch to spot overfitting

    # ── Evaluate ───────────────────────────────────────────────
       # Predicts values on the test set, converts them back to real units,
       # then calculates MAE, RMSE, and R² for temperature and humidity.
    y_pred_sc = model.predict(X_te, verbose=0)
    y_te_real = scaler.inverse_transform(y_te)
    y_pr_real = scaler.inverse_transform(y_pred_sc) 

    for i, feat in enumerate(["Temperature","Humidity"]):
        mae  = mean_absolute_error(y_te_real[:,i], y_pr_real[:,i])
        rmse = np.sqrt(mean_squared_error(y_te_real[:,i], y_pr_real[:,i]))
        rv   = r2(y_te_real[:,i], y_pr_real[:,i])
        unit = "°C" if i==0 else "%"
        print(f"  {feat:12s}  MAE={mae:.3f}{unit}  RMSE={rmse:.3f}  R²={rv:.3f}")

    # ── Plot ───────────────────────────────────────────────────
    #Creates graphs comparing actual vs predicted temperature and humidity.
    # The plots help visually show how close the model predictions are to real readings.
    
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(11,5),sharex=True)
    fig.suptitle(f"Room {room_id} ({name}) — LSTM Actual vs Predicted")
    for ax, idx, label, unit in [(ax1,0,"Temperature","°C"),(ax2,1,"Humidity","%")]:
        ax.plot(y_te_real[:,idx], label=f"Actual {label}")
        ax.plot(y_pr_real[:,idx], label=f"Predicted", linestyle="--", alpha=0.8)
        ax.set_ylabel(f"{label} ({unit})")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax2.set_xlabel("Test sample index")
    plt.tight_layout()
    png = f"room_{room_id}_lstm_eval.png"
    plt.savefig(png, dpi=130); plt.close()
    print(f"  Evaluation plot saved → {png}")

    # ── Extract weights ────────────────────────────────────────
    # Keras LSTM layer weights:
    #   [0] kernel          shape (input_size, 4*units)  = (2, 80)
    #   [1] recurrent_kernel shape (units, 4*units)       = (20,80)
    #   [2] bias             shape (4*units,) or (2,4*units)
    # Gets the trained LSTM and Dense layer weights.
    # These weights are needed so the trained model can be exported and used in n8n.
    # Stores room name, model settings, scaler values, last sequence,
    # date information, and all trained weights in a structured format.
    lstm_layer  = model.layers[0]
    dense_layer = model.layers[1]

    lw = lstm_layer.get_weights()
    dw = dense_layer.get_weights()

    bias = lw[2]
    if bias.ndim == 2:          # some Keras builds store (2, 4*units)
        bias = bias[0] + bias[1]

    # Flatten 2-D matrices to 1-D for compact JSON; JS reconstructs them
    def flat(arr): return [round(float(v), 6) for v in arr.flatten()]

    last_seq   = scaled[-SEQ_LEN:].tolist()
    last_date  = daily["timestamp"].iloc[-1].strftime("%Y-%m-%d")
    data_start = daily["timestamp"].iloc[0].strftime("%Y-%m-%d")

    return {
        "name":       name,
        "units":      int(UNITS),
        "seq_len":    int(SEQ_LEN),
        "scaler_min": [round(float(v),6) for v in scaler.data_min_],
        "scaler_max": [round(float(v),6) for v in scaler.data_max_],
        "last_seq":   [[round(float(v),6) for v in row] for row in last_seq],
        "last_date":  last_date,
        "data_start": data_start,
        # Weights — stored flat; JS reshapes using rows/cols from units
        "W_rows":  int(lw[0].shape[0]),   # input_size  (2)
        "W_cols":  int(lw[0].shape[1]),   # 4*units    (80)
        "W":       flat(lw[0]),           # kernel, shape (2,80)
        "U":       flat(lw[1]),           # recurrent_kernel, shape (20,80)
        "b":       flat(bias),            # bias, shape (80,)
        "Wd":      flat(dw[0]),           # dense kernel, shape (20,2)
        "bd":      flat(dw[1]),           # dense bias,   shape (2,)
    }


# ── MAIN ────────────────────────────────────────────────────────
# Controls the full workflow.
# It loads the dataset, trains the model for each room,
# saves all trained weights into one JSON file, and prints the next steps for n8n.
# Goes through each room ID, filters the dataset for that room,
# skips empty rooms, and trains a separate model for each available room.

def main():
    print("\n" + "═"*55)
    print("  LSTM Training — Smart Building")
    print("  One script. Two outputs: weights JSON + eval plots.")
    print("═"*55)

    df = load_data()

    export = {
        "generated_at": pd.Timestamp.now().isoformat()[:19],
        "rooms": {}
    }

    for rid in ROOM_IDS:
        df_room = df[df["room_id"] == rid].copy()
        if df_room.empty:
            print(f"\n  SKIP room {rid}: no rows in dataset")
            continue
        export["rooms"][str(rid)] = train_room(rid, df_room)

    if not export["rooms"]:
        raise RuntimeError("No rooms trained. Check room_id values in your CSV.")

    with open(OUT_FILE, "w") as f:
        json.dump(export, f, separators=(",",":"))   

    size_kb = os.path.getsize(OUT_FILE) / 1024
    print(f"\n{'═'*55}")
    print(f"  Done!  Weights saved → {OUT_FILE}  ({size_kb:.1f} KB)")
    print(f"  Rooms exported: {list(export['rooms'].keys())}")
    print(f"{'═'*55}")
    print()
    print("  NEXT STEP:")
    print(f"  1. Open {OUT_FILE} in a text editor")
    print( "  2. Copy ALL the text inside (Ctrl+A, Ctrl+C)")
    print( "  3. In n8n, open the LSTM Code node")
    print( "  4. Find the line that says:  PASTE_WEIGHTS_HERE")
    print( "  5. Replace that placeholder with the copied text")
    print( "  6. Save and Publish the workflow")
    print()


if __name__ == "__main__":
    main()
