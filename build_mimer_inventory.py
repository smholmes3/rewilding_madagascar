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

        # -------- Extract metadata from path --------
        # Adjust if needed depending on folder structure

        parts = full_path.split(os.sep)

        # Example assumptions:
        # .../scheduleA_raw_audio_2026-03/<recorder_id>/.../<filename>

        recorder_id = parts[-2]   # often correct, verify!

        filename = f

        # Optional: try to parse datetime from filename
        # Modify depending on naming convention

        datetime_start = ""
        date = ""

        # If filename contains timestamp like 20260315_083000.wav
        try:
            base = os.path.splitext(filename)[0]
            dt = datetime.strptime(base[:15], "%Y%m%d_%H%M%S")
            datetime_start = dt.isoformat()
            date = dt.date().isoformat()
        except:
            pass

        rows.append([
            full_path,
            filename,
            recorder_id,
            date,
            datetime_start
        ])

print(f"Collected {len(rows)} files")

with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "filepath",
        "filename",
        "recorder_id",
        "date",
        "datetime_start"
    ])
    writer.writerows(rows)

print(f"Saved inventory to {out_csv}")