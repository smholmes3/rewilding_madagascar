from pathlib import Path
import pandas as pd

IN_CSV = "data/allfiles_for_predicting_mac.csv"
OUT_DIR = Path("data/validation_outputs")
FILE_COL = "file"
CHECK_UNREADABLE = False  # start with False; True is slower

OUT_DIR.mkdir(parents=True, exist_ok=True)

def is_missing_or_zero(p: Path):
    try:
        if not p.exists():
            return True, "missing", ""
        if p.stat().st_size == 0:
            return True, "zero_bytes", ""
        return False, "", ""
    except OSError as e:
        return True, "os_error", repr(e)

def is_unreadable_soundfile(p: Path):
    try:
        import soundfile as sf
        _ = sf.info(str(p))
        return False, "", ""
    except Exception as e:
        return True, "unreadable", repr(e)

print("Loading allfiles...")
df = pd.read_csv(IN_CSV)
df[FILE_COL] = df[FILE_COL].astype(str)

unique_files = pd.Series(df[FILE_COL].unique(), name=FILE_COL)
bad_rows = []

print(f"Validating {len(unique_files):,} unique files...")

for i, fp in enumerate(unique_files, start=1):
    p = Path(fp)

    bad, reason, detail = is_missing_or_zero(p)
    if bad:
        bad_rows.append({"file": fp, "reason": reason, "detail": detail})
    elif CHECK_UNREADABLE:
        bad2, reason2, detail2 = is_unreadable_soundfile(p)
        if bad2:
            bad_rows.append({"file": fp, "reason": reason2, "detail": detail2})

    if i % 2000 == 0:
        print(f"  checked {i:,} files... bad so far: {len(bad_rows):,}")

bad_df = pd.DataFrame(bad_rows)
bad_path = OUT_DIR / "bad_files_allfiles.csv"
bad_df.to_csv(bad_path, index=False)

bad_set = set(bad_df["file"].tolist())
clean = df[~df[FILE_COL].isin(bad_set)].copy()

clean_path = OUT_DIR / "allfiles_for_predicting_mac_clean.csv"
clean.to_csv(clean_path, index=False)

print("Done.")
print(f"Rows in:  {len(df):,}")
print(f"Rows out: {len(clean):,}")
print(f"Bad files: {len(bad_df):,}")
print(f"Wrote: {bad_path}")
print(f"Wrote: {clean_path}")
