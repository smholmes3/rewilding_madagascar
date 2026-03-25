import argparse
import os
import pandas as pd

def load_inventory(inventory_csv, recorder_key):
    df = pd.read_csv(inventory_csv)
    sub = df[df["recorder_key"] == recorder_key].copy()
    if sub.empty:
        raise ValueError(f"No files found for recorder_key={recorder_key}")
    return sub.sort_values("datetime_start")

def run_predictions_for_recorder(sub_df):
    # placeholder:
    # load model
    # run predict on sub_df["filepath"]
    # return wide dataframe with metadata + species columns
    pass

def make_top20_review(pred_df, metadata_cols):
    species_cols = [c for c in pred_df.columns if c not in metadata_cols]

    long_df = pred_df.melt(
        id_vars=metadata_cols,
        value_vars=species_cols,
        var_name="species",
        value_name="score"
    )

    review_df = (
        long_df
        .sort_values(["date", "species", "score"], ascending=[True, True, False])
        .groupby(["date", "species"], group_keys=False)
        .head(20)
        .copy()
    )

    review_df["rank_within_day"] = (
        review_df
        .groupby(["date", "species"])["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    return review_df.sort_values(["date", "species", "rank_within_day"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory_csv", required=True)
    parser.add_argument("--recorder_key", required=True)
    parser.add_argument("--out_dir_predictions", required=True)
    parser.add_argument("--out_dir_review", required=True)
    args = parser.parse_args()

    sub_df = load_inventory(args.inventory_csv, args.recorder_key)

    pred_df = run_predictions_for_recorder(sub_df)

    metadata_cols = [
        "filepath", "filename", "site", "habitat_code",
        "recorder_id", "recorder_key", "date", "datetime_start"
    ]

    review_df = make_top20_review(pred_df, metadata_cols)

    os.makedirs(args.out_dir_predictions, exist_ok=True)
    os.makedirs(args.out_dir_review, exist_ok=True)

    pred_path = os.path.join(args.out_dir_predictions, f"predictions_{args.recorder_key}.csv")
    review_path = os.path.join(args.out_dir_review, f"top20_review_{args.recorder_key}.csv")

    pred_df.to_csv(pred_path, index=False)
    review_df.to_csv(review_path, index=False)

if __name__ == "__main__":
    main()