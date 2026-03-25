import os
import csv
from datetime import datetime

root = "/mimer/NOBACKUP/groups/rewilding_madagascar/scheduleA_raw_audio_2026-03/"
out_csv = "file_inventory_mimer.csv"

audio_exts = (".wav", ".WAV")

rows = []

for dirpath, _, filenames in os.walk(root):
    for f in filenames:
        if not f.endswith(audio_exts):
            continue

        full_path = os.path.join(dirpath, f)

        # ---------- Extract site from path ----------
        parts = full_path.split(os.sep)
        site = parts[-4]  # site/habitat/recorder/file

        # ---------- Parse filename ----------
        filename = f
        base = os.path.splitext(filename)[0]

        try:
            left, date_str, time_str = base.rsplit("_", 2)
            habitat_code, recorder_id = left.split("-", 1)

            dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
            datetime_start = dt.isoformat()
            date = dt.date().isoformat()

        except Exception as e:
            # Skip files that don't match expected pattern
            print(f"Skipping unexpected filename: {filename} ({e})")
            continue

        # ---------- Unique recorder key ----------
        recorder_key = f"{site}_{habitat_code}_{recorder_id}"

        rows.append([
            full_path,
            filename,
            site,
            habitat_code,
            recorder_id,
            recorder_key,
            date,
            datetime_start
        ])

print(f"Collected {len(rows)} files")

with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "filepath",
        "filename",
        "site",
        "habitat_code",
        "recorder_id",
        "recorder_key",
        "date",
        "datetime_start"
    ])
    writer.writerows(rows)

print(f"Saved inventory to {out_csv}")