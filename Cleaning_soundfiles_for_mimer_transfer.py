#Verify that I can capture schedule A files cleanly

import pandas as pd

df = pd.read_csv("data/allfiles_for_predicting.csv")
df["file"] = df["file"].astype(str)

print(df.head())
print(df.shape)

a = df[df["file"].str.contains(r"/[^/]+_A/", regex=True, na=False)].copy()
b = df[df["file"].str.contains(r"/[^/]+_B/", regex=True, na=False)].copy()

print("Rows with _A/:", len(a))
print("Rows with _B/:", len(b))

print("\nExample A paths:")
print(a["file"].drop_duplicates().head(10).to_list())

print("\nExample B paths:")
print(b["file"].drop_duplicates().head(10).to_list())


#Now I want a clean list of unique Schedule A files, and to check that they exist and are not zero bytes, so I can give that to Mimer for transfer. I'll also save a list of any bad files for review.
from pathlib import Path
import pandas as pd

IN_CSV = "data/allfiles_for_predicting.csv"
OUT_DIR = Path("data/validation_outputs")
FILE_COL = "file"

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

print("Loading file list...")
df = pd.read_csv(IN_CSV)
df[FILE_COL] = df[FILE_COL].astype(str)

# Keep only Schedule A using folder names ending in _A
df_a = df[df[FILE_COL].str.contains(r"/[^/]+_A/", regex=True, na=False)].copy()

# One row per actual source file
unique_files = pd.Series(df_a[FILE_COL].drop_duplicates(), name=FILE_COL)

bad_rows = []

print(f"Validating {len(unique_files):,} unique Schedule A files...")

for i, fp in enumerate(unique_files, start=1):
    p = Path(fp)

    bad, reason, detail = is_missing_or_zero(p)
    if bad:
        bad_rows.append({"file": fp, "reason": reason, "detail": detail})

    if i % 2000 == 0:
        print(f"  checked {i:,} files... bad so far: {len(bad_rows):,}")

bad_df = pd.DataFrame(bad_rows)
bad_path = OUT_DIR / "bad_files_scheduleA.csv"
bad_df.to_csv(bad_path, index=False)

bad_set = set(bad_df["file"].tolist())
clean_unique = unique_files[~unique_files.isin(bad_set)].copy()

clean_csv_path = OUT_DIR / "scheduleA_clean_files.csv"
clean_txt_path = OUT_DIR / "scheduleA_clean_paths.txt"

clean_unique.to_frame().to_csv(clean_csv_path, index=False)
clean_unique.to_csv(clean_txt_path, index=False, header=False)

print("Done.")
print(f"Schedule A unique files: {len(unique_files):,}")
print(f"Bad files: {len(bad_df):,}")
print(f"Clean files: {len(clean_unique):,}")
print(f"Wrote: {bad_path}")
print(f"Wrote: {clean_csv_path}")
print(f"Wrote: {clean_txt_path}")


#check files that were marked as bad
bad = pd.read_csv("data/validation_outputs/bad_files_scheduleA.csv")
print(bad["reason"].value_counts())
print(bad.head())