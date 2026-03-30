from pathlib import Path
import pandas as pd

in_path = Path("/mnt/d/rewilding_madagascar/data/metadata.csv")
out_path = Path("/mnt/d/data/metadata_mimer.csv")

metadata = pd.read_csv(in_path)

# Remove accidental index column if present
metadata = metadata.drop(columns=["Unnamed: 0"], errors="ignore")

OLD = "/mnt/class_data/group1_bioacoustics/sheila/"
NEW = "/mimer/NOBACKUP/groups/rewilding_madagascar/"

metadata["SoundFile_path"] = metadata["SoundFile_path"].str.replace(OLD, NEW, regex=False)
metadata["Raven_path"] = metadata["Raven_path"].str.replace(OLD, NEW, regex=False)

out_path.parent.mkdir(parents=True, exist_ok=True)
metadata.to_csv(out_path, index=False)

print(f"Saved updated metadata to: {out_path}")