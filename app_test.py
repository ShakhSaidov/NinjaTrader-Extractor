import argparse
import threading
import time

import pandas as pd
import pyarrow.parquet as pq
from lightweight_charts import Chart


# ============================================================
# PARQUET LOADING
# ============================================================

def get_parquet_range(path: str):
    """
    Get the first/last datetime directly from Parquet metadata/data
    without loading the entire file into pandas.
    """
    pf = pq.ParquetFile(path)

    first_dt = None
    last_dt = None

    # Only inspect the datetime column.
    for batch in pf.iter_batches(
        batch_size=1,
        columns=["datetime"]
    ):
        if batch.num_rows:
            first_dt = batch.column(0)[0].as_py()
            break

    # Read the final row group only.
    last_rg = pf.metadata.num_row_groups - 1

    table = pf.read_row_group(
        last_rg,
        columns=["datetime"]
    )

    if table.num_rows:
        last_dt = table["datetime"][-1].as_py()

    return pd.Timestamp(first_dt), pd.Timestamp(last_dt)


def load_1min(
    path: str,
    start_date=None,
    end_date=None,
    bars_before=5000,
):
    """
    Load ONLY the data needed by the replay chart.

    The parquet is assumed to already be chronologically ordered,
    which is the normal situation for your continuous NQ file.
    """

    columns = [
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    # --------------------------------------------------------
    # CASE 1:
    # No start date supplied.
    #
    # Load only the first bars_before rows rather than
    # loading 6.5 million rows.
    # --------------------------------------------------------

    if start_date is None:
        df = pd.read_parquet(
            path,
            columns=columns,
            engine="pyarrow"
        )

        df = df.iloc[:max(bars_before, 1)].copy()

    else:
        start = pd.Timestamp(start_date)

        # ----------------------------------------------------
        # We need some history BEFORE the requested date.
        #
        # Use a narrow date range first.
        # ----------------------------------------------------

        # Load the date range from the requested day onward.
        filters = [
            ("datetime", ">=", start.to_pydatetime())
        ]

        if end_date is not None:
            filters.append(
                ("datetime", "<", pd.Timestamp(end_date).to_pydatetime())
            )

        df = pd.read_parquet(
            path,
            columns=columns,
            filters=filters,
            engine="pyarrow"
        )

        # If there is no data, return cleanly.
        if df.empty:
            raise ValueError(
                f"No data found at or after {start_date}"
            )

        # ----------------------------------------------------
        # Add history before the requested date.
        #
        # We only need bars_before rows, not millions of rows.
        # ----------------------------------------------------

        before = pd.read_parquet(
            path,
            columns=columns,
            filters=[
                ("datetime", "<", start.to_pydatetime())
            ],
            engine="pyarrow"
        )

        if not before.empty:
            before = before.tail(bars_before)

            df = pd.concat(
                [before, df],
                ignore_index=True
            )

    # --------------------------------------------------------
    # Datetime handling
    # --------------------------------------------------------
    #
    # DO NOT do:
    #
    # astype("int64") // 10**9
    #
    # on datetime64[us].
    #
    # The previous 1970 problem came from treating microseconds
    # as nanoseconds.
    #
    # Pandas can safely convert this to datetime64[ns] here,
    # but we're doing it only on the reduced dataframe.
    # --------------------------------------------------------

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    ).astype("datetime64[ns]")

    # --------------------------------------------------------
    # Rename for lightweight_charts
    # --------------------------------------------------------

    df = df.rename(
        columns={"datetime": "date"}
    )

    df = df[
        ["date", "open", "high", "low", "close", "volume"]
    ]

    # --------------------------------------------------------
    # We deliberately DO NOT:
    #
    # sort_values()
    # drop_duplicates()
    #
    # because your continuous parquet is already chronological.
    #
    # Those operations are expensive on millions of rows.
    # --------------------------------------------------------

    return df.reset_index(drop=True)


# ============================================================
# REPLAY APPLICATION
# ============================================================

class ReplayApp:

    def __init__(
        self,
        df: pd.DataFrame,
        chart: Chart,
        start_bars: int = 5000
    ):
        self.df = df
        self.chart = chart

        self.playing = False
        self.speed_bars_per_sec = 5

        self.i = min(
            start_bars,
            len(df)
        )

        self._lock = threading.Lock()

    # --------------------------------------------------------
    # JUMP
    # --------------------------------------------------------

    def jump_to_date(self, date_str: str):

        target = pd.Timestamp(date_str)

        pos = self.df["date"].searchsorted(target)

        pos = max(pos, 1)

        self.i = pos

        self.chart.set(
            self.df.iloc[:pos]
        )

        print(
            f"Jumped to {date_str} -> "
            f"row {pos:,} "
            f"({self.df['date'].iloc[pos - 1]})"
        )

    # --------------------------------------------------------
    # ONE BAR
    # --------------------------------------------------------

    def step_one(self):

        if self.i >= len(self.df):
            return

        row = self.df.iloc[self.i]

        self.chart.update(row)

        self.i += 1

    # --------------------------------------------------------
    # PLAY LOOP
    # --------------------------------------------------------

    def play_loop(self):

        while True:

            with self._lock:
                playing = self.playing
                speed = self.speed_bars_per_sec

            if playing:

                self.step_one()

                time.sleep(
                    max(
                        1.0 / max(speed, 1),
                        0.01
                    )
                )

            else:
                time.sleep(0.1)

    # --------------------------------------------------------
    # PLAY / PAUSE
    # --------------------------------------------------------

    def toggle_play(self, chart):

        with self._lock:
            self.playing = not self.playing

        print(
            "Playing"
            if self.playing
            else
            "Paused"
        )

    # --------------------------------------------------------
    # JUMP TEXTBOX
    # --------------------------------------------------------

    def on_jump_textbox(self, chart):

        val = chart.topbar["jump_date"].value

        try:
            self.jump_to_date(val)

        except Exception as e:
            print(
                f"[jump error] '{val}': {e}"
            )

    # --------------------------------------------------------
    # STEP BUTTON
    # --------------------------------------------------------

    def step_once_button(self, chart):

        with self._lock:
            self.playing = False

        self.step_one()


# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--parquet",
        required=True
    )

    ap.add_argument(
        "--start",
        default=None,
        help="YYYY-MM-DD"
    )

    ap.add_argument(
        "--end",
        default=None,
        help="YYYY-MM-DD"
    )

    ap.add_argument(
        "--start-bars",
        type=int,
        default=5000,
        help="bars of history before replay start"
    )

    args = ap.parse_args()

    print()
    print(
        f"Loading requested data from:\n"
        f"{args.parquet}"
    )

    start_time = time.perf_counter()

    df = load_1min(
        args.parquet,
        start_date=args.start,
        end_date=args.end,
        bars_before=args.start_bars
    )

    elapsed = time.perf_counter() - start_time

    print(
        f"Loaded {len(df):,} rows "
        f"in {elapsed:.2f} seconds"
    )

    print(
        f"Range: "
        f"[{df['date'].iloc[0]} .. "
        f"{df['date'].iloc[-1]}]"
    )

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    chart = Chart(
        toolbox=True
    )

    chart.legend(
        visible=True
    )

    app = ReplayApp(
        df,
        chart,
        start_bars=args.start_bars
    )

    # --------------------------------------------------------
    # TOPBAR
    # --------------------------------------------------------

    default_date = (
        args.start
        if args.start
        else str(df["date"].iloc[0].date())
    )

    chart.topbar.textbox(
        "jump_date",
        default_date
    )

    chart.topbar.button(
        "jump_btn",
        "Jump",
        func=app.on_jump_textbox
    )

    chart.topbar.button(
        "play_btn",
        "Play/Pause",
        func=app.toggle_play
    )

    chart.topbar.button(
        "step_btn",
        "+1 Bar",
        func=app.step_once_button
    )

    # --------------------------------------------------------
    # INITIAL DATA
    # --------------------------------------------------------

    if args.start:

        app.jump_to_date(
            args.start
        )

    else:

        initial = min(
            args.start_bars,
            len(df)
        )

        chart.set(
            df.iloc[:initial]
        )

        app.i = initial

    # --------------------------------------------------------
    # REPLAY THREAD
    # --------------------------------------------------------

    t = threading.Thread(
        target=app.play_loop,
        daemon=True
    )

    t.start()

    print("Starting chart...")

    chart.show(
        block=True
    )


if __name__ == "__main__":
    main()