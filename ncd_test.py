import struct
import sys
from datetime import datetime, timedelta
from pathlib import Path


# ------------------------------------------------------------
# Binary helpers
# NinjaTrader NCD header uses little-endian values.
# Record data uses big-endian integer deltas.
# ------------------------------------------------------------

def u8(data, pos):
    return data[pos], pos + 1


def u16be(data, pos):
    return struct.unpack_from(">H", data, pos)[0], pos + 2


def u32be(data, pos):
    return struct.unpack_from(">I", data, pos)[0], pos + 4


def u64be(data, pos):
    return struct.unpack_from(">Q", data, pos)[0], pos + 8


def dotnet_datetime(ticks):
    """
    Convert .NET DateTime ticks since 0001-01-01 into Python datetime.
    1 tick = 100 ns.
    """
    microseconds = ticks // 10
    return datetime(1, 1, 1) + timedelta(microseconds=microseconds)


# ------------------------------------------------------------
# Minute NCD reader
# Based directly on the publicly reverse-engineered
# NinjaTraderNCDFiles NCDMinuteFile implementation.
# ------------------------------------------------------------

def read_ncd_minute(path):
    path = Path(path)
    data = path.read_bytes()

    if len(data) < 28:
        raise ValueError(f"File is too small to be a valid NCD file: {path}")

    pos = 0

    # NCD header
    tick_size_time = struct.unpack_from("<I", data, pos)[0]
    pos += 4

    tick_size_price = struct.unpack_from("<d", data, pos)[0]
    pos += 8

    last_price = struct.unpack_from("<d", data, pos)[0]
    pos += 8

    file_start_ticks = struct.unpack_from("<Q", data, pos)[0]
    pos += 8

    file_start_datetime = dotnet_datetime(file_start_ticks)

    records = []

    while pos < len(data):

        # Need at least the two mask bytes
        if pos + 2 > len(data):
            break

        mask1 = data[pos]
        mask2 = data[pos + 1]
        pos += 2

        # ----------------------------------------------------
        # Time delta
        #
        # MinuteMask1:
        # Time1 = 1
        # Time2 = 2
        # Time4 = 3
        # ----------------------------------------------------

        time_flag = mask1 & 0x03

        if time_flag == 3:
            time_delta, pos = u32be(data, pos)
        elif time_flag == 1:
            time_delta, pos = u8(data, pos)
        elif time_flag == 2:
            time_delta, pos = u16be(data, pos)
        else:
            time_delta = 1

        bar_datetime = last_datetime + timedelta(minutes=time_delta) \
            if records else file_start_datetime + timedelta(minutes=time_delta)

        # ----------------------------------------------------
        # Open
        #
        # Open1 = 4
        # Open2 = 8
        # Open4 = 12
        # ----------------------------------------------------

        open_flag = mask1 & 0x0C

        if open_flag == 12:
            open_delta, pos = u32be(data, pos)
            open_offset = 1073741824
        elif open_flag == 4:
            open_delta, pos = u8(data, pos)
            open_offset = 128
        elif open_flag == 8:
            open_delta, pos = u16be(data, pos)
            open_offset = 32768
        else:
            open_delta = 0
            open_offset = 0

        bar_open = round(
            last_price +
            ((open_delta - open_offset) * tick_size_price),
            2
        )

        # ----------------------------------------------------
        # High
        #
        # High1 = 16
        # High2 = 32
        # High4 = 48
        # ----------------------------------------------------

        high_flag = mask2 & 0x30

        if high_flag == 48:
            high_delta, pos = u32be(data, pos)
        elif high_flag == 16:
            high_delta, pos = u8(data, pos)
        elif high_flag == 32:
            high_delta, pos = u16be(data, pos)
        else:
            high_delta = 0

        bar_high = round(
            bar_open + (high_delta * tick_size_price),
            2
        )

        # ----------------------------------------------------
        # Low
        #
        # Low1 = 64
        # Low2 = 128
        # Low4 = 192
        # ----------------------------------------------------

        low_flag = mask2 & 0xC0

        if low_flag == 192:
            low_delta, pos = u32be(data, pos)
        elif low_flag == 64:
            low_delta, pos = u8(data, pos)
        elif low_flag == 128:
            low_delta, pos = u16be(data, pos)
        else:
            low_delta = 0

        bar_low = round(
            bar_open - (low_delta * tick_size_price),
            2
        )

        # ----------------------------------------------------
        # Close
        #
        # Close1 = 1
        # Close2 = 2
        # Close4 = 3
        # ----------------------------------------------------

        close_flag = mask2 & 0x03

        if close_flag == 3:
            close_delta, pos = u32be(data, pos)
        elif close_flag == 1:
            close_delta, pos = u8(data, pos)
        elif close_flag == 2:
            close_delta, pos = u16be(data, pos)
        else:
            close_delta = 0

        bar_close = round(
            bar_low + (close_delta * tick_size_price),
            2
        )

        # ----------------------------------------------------
        # Volume
        #
        # Volume8  = 240
        # Volume4  = 192
        # Volume2  = 160
        # Volume1X1000 = 128
        # Volume1X500  = 96
        # Volume1       = 32
        # Volume1X100   = 64
        # ----------------------------------------------------

        volume_flag = mask1 & 0xE0

        if volume_flag == 240:
            volume, pos = u64be(data, pos)
            multiplier = 1
        elif volume_flag == 192:
            volume, pos = u32be(data, pos)
            multiplier = 1
        elif volume_flag == 160:
            volume, pos = u16be(data, pos)
            multiplier = 1
        elif volume_flag == 128:
            volume, pos = u8(data, pos)
            multiplier = 1000
        elif volume_flag == 96:
            volume, pos = u8(data, pos)
            multiplier = 500
        elif volume_flag == 32:
            volume, pos = u8(data, pos)
            multiplier = 1
        elif volume_flag == 64:
            volume, pos = u8(data, pos)
            multiplier = 100
        else:
            raise ValueError(
                f"Unknown volume flag {volume_flag} "
                f"at byte position {pos} in {path}"
            )

        bar_volume = volume * multiplier

        records.append({
            "datetime": bar_datetime,
            "open": bar_open,
            "high": bar_high,
            "low": bar_low,
            "close": bar_close,
            "volume": bar_volume,
        })

        last_datetime = bar_datetime

        # IMPORTANT:
        # NinjaTrader NCD minute decoder uses the OPEN
        # as the previous price for the next bar.
        last_price = bar_open

    return {
        "file": str(path),
        "bytes": len(data),
        "tick_size_time": tick_size_time,
        "tick_size_price": tick_size_price,
        "file_start_datetime": file_start_datetime,
        "file_start_price": struct.unpack_from("<d", data, 12)[0],
        "records": records,
    }


# ------------------------------------------------------------
# Main test
# ------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print(r'python ncd_test.py "FULL_PATH_TO_NCD_FILE"')
        sys.exit(1)

    file_path = sys.argv[1]

    print()
    print("Reading:")
    print(file_path)
    print()

    result = read_ncd_minute(file_path)

    print("SUCCESS: NCD file decoded")
    print()
    print(f"File size:       {result['bytes']:,} bytes")
    print(f"Tick size:       {result['tick_size_price']}")
    print(f"File start time: {result['file_start_datetime']}")
    print(f"Start price:     {result['file_start_price']}")
    print(f"Records:         {len(result['records']):,}")
    print()

    print("First 10 records:")
    print("-" * 85)

    for r in result["records"][:10]:
        print(
            f"{r['datetime']}  "
            f"O={r['open']:<10} "
            f"H={r['high']:<10} "
            f"L={r['low']:<10} "
            f"C={r['close']:<10} "
            f"V={r['volume']}"
        )

    print()
    print("Last 5 records:")
    print("-" * 85)

    for r in result["records"][-5:]:
        print(
            f"{r['datetime']}  "
            f"O={r['open']:<10} "
            f"H={r['high']:<10} "
            f"L={r['low']:<10} "
            f"C={r['close']:<10} "
            f"V={r['volume']}"
        )