from __future__ import annotations

import argparse
import re
import struct
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Default path to your NinjaTrader DB minute folder
DEFAULT_SOURCE = Path(r"C:\Users\shakh\Documents\NinjaTrader 8\db\minute")

# ============================================================
# NCD BINARY READER
# ============================================================

def u8(data, pos): 
    return data[pos], pos + 1

def u16be(data, pos): 
    return struct.unpack_from(">H", data, pos)[0], pos + 2

def u32be(data, pos): 
    return struct.unpack_from(">I", data, pos)[0], pos + 4

def u64be(data, pos): 
    return struct.unpack_from(">Q", data, pos)[0], pos + 8

def dotnet_datetime(ticks):
    """Convert .NET DateTime ticks to Python datetime."""
    microseconds = ticks // 10
    return datetime(1, 1, 1) + timedelta(microseconds=microseconds)

def read_ncd_minute(path: Path) -> pd.DataFrame:
    """Decodes a single NinjaTrader .ncd daily minute file into a DataFrame."""
    data = path.read_bytes()
    if len(data) < 28:
        raise ValueError(f"{path} is too small to be a valid NCD file")

    pos = 0
    tick_size_time = struct.unpack_from("<I", data, pos)[0]; pos += 4
    tick_size_price = struct.unpack_from("<d", data, pos)[0]; pos += 8
    start_price = struct.unpack_from("<d", data, pos)[0]; pos += 8
    file_start_ticks = struct.unpack_from("<Q", data, pos)[0]; pos += 8
    file_start_datetime = dotnet_datetime(file_start_ticks)

    records = []
    last_datetime = None
    last_price = start_price

    while pos < len(data):
        if pos + 2 > len(data):
            break

        mask1, mask2 = data[pos], data[pos + 1]
        pos += 2

        # TIME
        time_flag = mask1 & 0x03
        if time_flag == 3: time_delta, pos = u32be(data, pos)
        elif time_flag == 1: time_delta, pos = u8(data, pos)
        elif time_flag == 2: time_delta, pos = u16be(data, pos)
        else: time_delta = 1

        bar_datetime = file_start_datetime + timedelta(minutes=time_delta) if last_datetime is None else last_datetime + timedelta(minutes=time_delta)

        # OPEN
        open_flag = mask1 & 0x0C
        if open_flag == 12: open_delta, pos, open_offset = *u32be(data, pos), 1073741824
        elif open_flag == 4: open_delta, pos, open_offset = *u8(data, pos), 128
        elif open_flag == 8: open_delta, pos, open_offset = *u16be(data, pos), 32768
        else: open_delta, open_offset = 0, 0
        
        bar_open = round(last_price + ((open_delta - open_offset) * tick_size_price), 10)

        # HIGH
        high_flag = mask2 & 0x30
        if high_flag == 48: high_delta, pos = u32be(data, pos)
        elif high_flag == 16: high_delta, pos = u8(data, pos)
        elif high_flag == 32: high_delta, pos = u16be(data, pos)
        else: high_delta = 0
        
        bar_high = round(bar_open + high_delta * tick_size_price, 10)

        # LOW
        low_flag = mask2 & 0xC0
        if low_flag == 192: low_delta, pos = u32be(data, pos)
        elif low_flag == 64: low_delta, pos = u8(data, pos)
        elif low_flag == 128: low_delta, pos = u16be(data, pos)
        else: low_delta = 0
        
        bar_low = round(bar_open - low_delta * tick_size_price, 10)

        # CLOSE
        close_flag = mask2 & 0x03
        if close_flag == 3: close_delta, pos = u32be(data, pos)
        elif close_flag == 1: close_delta, pos = u8(data, pos)
        elif close_flag == 2: close_delta, pos = u16be(data, pos)
        else: close_delta = 0
        
        bar_close = round(bar_low + close_delta * tick_size_price, 10)

        # VOLUME
        volume_flag = mask1 & 0xE0
        if volume_flag == 240: volume, pos, multiplier = *u64be(data, pos), 1
        elif volume_flag == 192: volume, pos, multiplier = *u32be(data, pos), 1
        elif volume_flag == 160: volume, pos, multiplier = *u16be(data, pos), 1
        elif volume_flag == 128: volume, pos, multiplier = *u8(data, pos), 1000
        elif volume_flag == 96: volume, pos, multiplier = *u8(data, pos), 500
        elif volume_flag == 32: volume, pos, multiplier = *u8(data, pos), 1
        elif volume_flag == 64: volume, pos, multiplier = *u8(data, pos), 100
        else: raise ValueError(f"Unknown volume flag {volume_flag} at byte {pos} in {path}")
        
        bar_volume = int(volume * multiplier)

        records.append((bar_datetime, bar_open, bar_high, bar_low, bar_close, bar_volume))
        last_datetime = bar_datetime
        last_price = bar_open

    return pd.DataFrame(records, columns=["datetime", "open", "high", "low", "close", "volume"])


# ============================================================
# CONTRACT DISCOVERY & PARSING
# ============================================================

def parse_contract_name(name: str):
    """Parses folder names like 'FDAX 03-20' or 'NQ 12-25'."""
    match = re.match(r"^(.+?)\s+(\d{2})-(\d{2})$", name)
    if not match: return None
    instrument = match.group(1).strip()
    month = int(match.group(2))
    year = 2000 + int(match.group(3))
    
    if not 1 <= month <= 12: return None
    return {"instrument": instrument, "month": month, "year": year, "name": name}

def discover_contracts(source: Path, instrument: str):
    contracts = []
    for directory in source.iterdir():
        if not directory.is_dir(): continue
        parsed = parse_contract_name(directory.name)
        if parsed is None or parsed["instrument"].upper() != instrument.upper(): continue
        
        files = [f for f in sorted(directory.glob("*.Last.ncd")) if re.match(r"^\d{8}\.Last\.ncd$", f.name)]
        parsed["path"] = directory
        parsed["files"] = files
        contracts.append(parsed)
        
    # Sort chronologically by year, then month
    contracts.sort(key=lambda x: (x["year"], x["month"]))
    return contracts


# ============================================================
# EXTRACT ONE CONTRACT (Saves to Parquet)
# ============================================================

def extract_contract(contract_info, output_dir):
    contract = contract_info["name"]
    files = contract_info["files"]
    
    contracts_dir = output_dir / "contracts"
    safe_name = contract.replace(" ", "_")
    output_file = contracts_dir / f"{safe_name}.parquet"

    print(f"\n{'='*70}\nExtracting: {contract}\nNCD files:  {len(files)}\n{'='*70}")

    if not files:
        return {"status": "EMPTY"}, []

    frames = []
    errors = []
    total_files = len(files)

    for i, file in enumerate(files, 1):
        try:
            df = read_ncd_minute(file)
            if not df.empty: frames.append(df)
            if i == 1 or i == total_files or i % 25 == 0:
                print(f"  {i:>4}/{total_files} {file.name} | {len(df):>6} bars")
        except Exception as e:
            errors.append({"contract": contract, "file": str(file), "error": str(e)})
            print(f"  ERROR: {file.name}: {e}")

    if not frames:
        return {"status": "FAILED"}, errors

    # Combine, deduplicate, and sort
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["datetime"], keep="last").sort_values("datetime").reset_index(drop=True)
    
    # ==============================================================
    # 1. STANDARDIZE TIMESTAMP: Convert 'Bar Close' to 'Bar Open'
    # ==============================================================
    df['datetime'] = df['datetime'] - pd.Timedelta(minutes=1)

    # ==============================================================
    # 2. TIMEZONE FIX: Convert local Helsinki time to pure UTC
    # ==============================================================
    df['datetime'] = (
        df['datetime']
        .dt.tz_localize("Europe/Helsinki", ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )
    # ==============================================================

    df.insert(1, "contract", contract)

    contracts_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False, compression="zstd")
    
    print(f"\n  Saved:  {output_file}\n  Bars:   {len(df):,}\n  Status: PASS")
    return {"status": "PASS"}, errors


# ============================================================
# LOAD CONTRACT PARQUETS
# ============================================================

def load_contract_parquets(output_dir, instrument):
    contracts_dir = output_dir / "contracts"
    files = sorted(contracts_dir.glob("*.parquet"))
    result = {}

    for file in files:
        try:
            df = pd.read_parquet(file)
            if df.empty: continue
            contract = str(df["contract"].iloc[0])
            parsed = parse_contract_name(contract)
            if parsed is None or parsed["instrument"].upper() != instrument.upper(): continue
            result[contract] = df
        except Exception as e:
            print(f"WARNING: could not load {file}: {e}")

    # Re-sort loaded data chronologically to be absolutely safe
    return dict(sorted(result.items(), key=lambda item: (parse_contract_name(item[0])["year"], parse_contract_name(item[0])["month"])))


# ============================================================
# FOOLPROOF ROLLOVER DETECTION
# ============================================================

def detect_rollover(old_df, new_df, old_contract, new_contract):
    """
    Robust 3-tier rollover detection. Guaranteed to find a transition date.
    """
    old_daily = old_df.set_index("datetime")["volume"].resample("1D").sum()
    new_daily = new_df.set_index("datetime")["volume"].resample("1D").sum()

    combined = pd.concat([old_daily.rename("old_vol"), new_daily.rename("new_vol")], axis=1).fillna(0)
    
    # Filter only to days where the new contract actually exists
    new_active = combined[combined["new_vol"] > 0]
    if new_active.empty:
        return None

    overlap = combined[(combined["old_vol"] > 0) & (combined["new_vol"] > 0)]
    crossover_days = overlap[overlap["new_vol"] > overlap["old_vol"]]

    if not crossover_days.empty:
        # TIER 1: First day where New Contract Volume > Old Contract Volume
        roll_date = crossover_days.index[0]
        method = "Volume Crossover"
    elif not overlap.empty:
        # TIER 2: Overlap exists, but old volume stayed higher. Roll on the last shared day.
        roll_date = overlap.index[-1]
        method = "Last Overlap Day"
    else:
        # TIER 3: No overlapping days at all. Roll on the first day the new contract appears.
        roll_date = new_active.index[0]
        method = "First Available (No Overlap)"

    # Calculate adjustment gap: Gap = New Price - Old Price
    # Old Price = Last close before midnight of the roll date
    old_before = old_df[old_df["datetime"] < roll_date]
    old_price = float(old_before.iloc[-1]["close"]) if not old_before.empty else float(old_df.iloc[-1]["close"])

    # New Price = First open on or after midnight of the roll date
    new_after = new_df[new_df["datetime"] >= roll_date]
    new_price = float(new_after.iloc[0]["open"]) if not new_after.empty else float(new_df.iloc[0]["open"])

    adjustment = new_price - old_price

    return {
        "old_contract": old_contract,
        "new_contract": new_contract,
        "roll_date": roll_date,
        "old_price": round(old_price, 4),
        "new_price": round(new_price, 4),
        "adjustment": round(adjustment, 4),
        "method": method
    }


# ============================================================
# BUILD CONTINUOUS & BACK-ADJUSTED SERIES
# ============================================================

def build_continuous(contract_data, output_dir, instrument):
    continuous_dir = output_dir / "continuous"
    reports_dir = output_dir / "reports"
    continuous_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    contracts = list(contract_data.keys())
    if not contracts:
        return

    print(f"\n{'='*70}\nDETECTING {instrument.upper()} ROLLOVERS\n{'='*70}")

    rollover_by_pair = {}
    rollover_rows = []

    # Step 1: Detect all rollovers between chronological pairs
    for old_contract, new_contract in zip(contracts, contracts[1:]):
        result = detect_rollover(
            contract_data[old_contract], contract_data[new_contract], 
            old_contract, new_contract
        )
        rollover_by_pair[(old_contract, new_contract)] = result
        
        if result:
            print(f"{old_contract} -> {new_contract}: Roll on {result['roll_date'].date()} | Gap: {result['adjustment']:>8.2f} | Method: {result['method']}")
            rollover_rows.append(result)

    pd.DataFrame(rollover_rows).to_csv(reports_dir / f"{instrument}_rollover_table.csv", index=False)

    # Step 2: Build RAW Continuous (Stitching actual prices)
    raw_parts = []
    for i, contract in enumerate(contracts):
        df = contract_data[contract].copy()
        
        # Chop off everything before this contract's roll-in date
        if i > 0:
            prev_roll = rollover_by_pair.get((contracts[i-1], contract))
            if prev_roll: df = df[df["datetime"] >= pd.Timestamp(prev_roll["roll_date"])]
            
        # Chop off everything after this contract's roll-out date
        if i < len(contracts) - 1:
            next_roll = rollover_by_pair.get((contract, contracts[i+1]))
            if next_roll: df = df[df["datetime"] < pd.Timestamp(next_roll["roll_date"])]
            
        if not df.empty:
            raw_parts.append(df)

    raw = pd.concat(raw_parts, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    raw_path = continuous_dir / f"{instrument}_continuous_raw.parquet"
    raw.to_parquet(raw_path, index=False, compression="zstd")

    # Step 3: Build BACK-ADJUSTED Continuous
    # We walk backwards from the newest contract to the oldest, ADDING the gaps so everything aligns smoothly.
    adjusted_parts = []
    cumulative_adjustment = 0.0

    for i in reversed(range(len(contracts))):
        contract = contracts[i]
        df = contract_data[contract].copy()

        # Same date slicing logic as RAW
        if i > 0:
            prev_roll = rollover_by_pair.get((contracts[i-1], contract))
            if prev_roll: df = df[df["datetime"] >= pd.Timestamp(prev_roll["roll_date"])]
        if i < len(contracts) - 1:
            next_roll = rollover_by_pair.get((contract, contracts[i+1]))
            if next_roll: df = df[df["datetime"] < pd.Timestamp(next_roll["roll_date"])]

        if df.empty: continue

        # Apply cumulative adjustment to older contracts to close the gap seamlessly
        for col in ["open", "high", "low", "close"]:
            df[col] = round(df[col] + cumulative_adjustment, 6)
            
        adjusted_parts.append(df)

        # Accumulate the NEXT gap for the older contracts we are about to process
        if i > 0:
            prev_roll = rollover_by_pair.get((contracts[i-1], contract))
            if prev_roll: cumulative_adjustment += prev_roll["adjustment"]

    adjusted = pd.concat(adjusted_parts, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    adjusted_path = continuous_dir / f"{instrument}_continuous_backadjusted.parquet"
    adjusted.to_parquet(adjusted_path, index=False, compression="zstd")

    print(f"\n{'='*70}\n{instrument.upper()} CONTINUOUS DATA BUILT SUCCESSFULLY\n{'='*70}")
    print(f"Total Contracts Stitched: {len(contracts)}")
    print(f"Rollovers Found:          {len(rollover_rows)}")
    print(f"Raw Output:               {raw_path} ({len(raw):,} bars)")
    print(f"Adjusted Output:          {adjusted_path} ({len(adjusted):,} bars)")
    print(f"Rollover Report CSV:      {reports_dir / f'{instrument}_rollover_table.csv'}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Universal NinjaTrader NCD to Continuous Futures Converter")
    parser.add_argument("--instrument", required=True, help="E.g., FDAX, NQ, ES, CL, GC")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Path to NT8 db/minute folder")
    parser.add_argument("--output", type=Path, default=None, help="Output directory folder")
    args = parser.parse_args()

    instrument = args.instrument.strip()
    source = args.source
    output = args.output if args.output else Path(f"{instrument.upper()}_1min_output")

    if not source.exists():
        sys.exit(f"ERROR: Source directory {source} does not exist.")
    
    output.mkdir(parents=True, exist_ok=True)
    contracts = discover_contracts(source, instrument)

    if not contracts:
        sys.exit(f"ERROR: No contracts found for {instrument}. Make sure the symbol matches the folder prefixes in {source}.")

    # 1. Force Extract Raw Data (Guarantees no skipped data updates)
    for contract_info in contracts:
        extract_contract(contract_info, output)

    # 2. Load newly extracted Parquets in chronological order
    contract_data = load_contract_parquets(output, instrument)
    
    # 3. Stitch and build Continuous / Back-Adjusted datasets
    build_continuous(contract_data, output, instrument)

if __name__ == "__main__":
    main()