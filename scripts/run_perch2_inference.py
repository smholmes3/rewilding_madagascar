import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import bioacoustics_model_zoo as bmz


class_list = [
    'Hypsipetes_madagascariensis','Copsychus_albospecularis','Coracopsis_nigra','Dicrurus_forficatus',
    'Coua_caerulea','Zosterops_maderaspatanus','Eurystomus_glaucurus','Agapornis_canus','Saxicola_torquatus',
    'Cyanolanius_madagascarinus','Leptopterus_chabert','Nesoenas_picturatus','Coua_reynaudii',
    'Ceblepyris_cinereus','Neodrepanis_coruscans','Philepitta_castanea','Eulemur_sp','Coua_cristata',
    'Treron_australis'
]


def load_model(state_dict_path):
    perch2_model = bmz.Perch2()
    perch2_model.initialize_custom_classifier(class_list, hidden_layer_sizes=())

    sd = torch.load(state_dict_path, weights_only=False)
    perch2_model.network.load_state_dict(sd)
    perch2_model.network.eval()

    print(f"Loaded classifier weights from: {state_dict_path}")
    return perch2_model


def make_top20_review(full_df):
    metadata_cols = [
        "filepath", "filename", "site", "habitat_code", "recorder_id",
        "recorder_key", "date", "datetime_start", "start_time", "end_time"
    ]

    long_df = full_df.melt(
        id_vars=metadata_cols,
        value_vars=class_list,
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
        review_df.groupby(["date", "species"])["score"]
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
    parser.add_argument(
        "--state_dict_path",
        default="/mimer/NOBACKUP/groups/rewilding_madagascar/models/perch2_shallow_classifier/classifier_state_dict.pt"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    inventory = pd.read_csv(args.inventory_csv)
    sub_df = inventory[inventory["recorder_key"] == args.recorder_key].copy()

    if sub_df.empty:
        raise ValueError(f"No files found for recorder_key={args.recorder_key}")

    sub_df = sub_df.sort_values("datetime_start").reset_index(drop=True)
    files = sub_df["filepath"].tolist()

    print(f"Recorder: {args.recorder_key}")
    print(f"Number of files: {len(files)}")

    model = load_model(args.state_dict_path)

    preds = model.predict(files, batch_size=args.batch_size)
    print("Prediction index names:", preds.index.names)
    print("Prediction columns (first 10):", list(preds.columns)[:10])

    preds = preds[class_list].copy()
    preds = preds.clip(-50, 50)
    preds = 1 / (1 + np.exp(-preds))
    preds = preds.reset_index()

    # expected from OpenSoundscape: file, start_time, end_time
    preds = preds.rename(columns={"file": "filepath"})

    full_df = preds.merge(
        sub_df,
        on="filepath",
        how="left"
    )

    out_pred_dir = Path(args.out_dir_predictions)
    out_review_dir = Path(args.out_dir_review)
    out_pred_dir.mkdir(parents=True, exist_ok=True)
    out_review_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_pred_dir / f"predictions_{args.recorder_key}.csv"
    full_df.to_csv(pred_path, index=False)
    print(f"Wrote predictions: {pred_path}")

    review_df = make_top20_review(full_df)
    review_path = out_review_dir / f"top20_review_{args.recorder_key}.csv"
    review_df.to_csv(review_path, index=False)
    print(f"Wrote review table: {review_path}")


if __name__ == "__main__":
    main()