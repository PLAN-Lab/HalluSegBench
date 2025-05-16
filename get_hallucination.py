import os
import cv2
import json
import argparse
import numpy as np

from tqdm import tqdm

def compute_score(mask_gt, mask_pred, image_shape):
    mask_gt = mask_gt.astype(np.uint8)
    mask_pred = mask_pred.astype(np.uint8)

    mask_pred = cv2.resize(mask_pred, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_gt = cv2.resize(mask_gt, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_gt = mask_gt.astype(bool)
    mask_pred = mask_pred.astype(bool)

    alpha = 3
    intersection = np.logical_and(mask_gt, mask_pred).sum()
    b_outside_a = np.logical_and(mask_pred, np.logical_not(mask_gt)).sum()
    denom = alpha * mask_gt.sum()
    numerator = alpha * intersection + b_outside_a

    return numerator / denom if denom > 0 else 0.0


def main(args):
    
    json_path = args.json_path
    data_base_path = args.base_path

    with open(json_path, "r") as f:
        data = json.load(f)

    for item in tqdm(data, desc="Calculating hallucination scores"):
        mask_file = os.path.join(data_base_path, item["mask_file"])
        pred_path = item.get("mask_edtl_orgi")

        if pred_path and os.path.exists(mask_file) and os.path.exists(pred_path):
            mask_gt = cv2.imread(mask_file, 0)
            mask_pred = cv2.imread(pred_path, 0)
            score1 = compute_score(mask_gt, mask_pred, mask_gt.shape)
            item["score_edtl_orgi"] = round(score1, 4)

        new_mask_file = os.path.join(data_base_path, item["new_mask_path"])
        pred_path = item.get("mask_orgl_edti")

        if pred_path and os.path.exists(new_mask_file) and os.path.exists(pred_path):
            mask_gt = cv2.imread(new_mask_file, 0)
            mask_pred = cv2.imread(pred_path, 0)
            score2 = compute_score(mask_gt, mask_pred, mask_gt.shape)
            item["score_orgl_edti"] = round(score2, 4)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Scores computed and updated json saved to {json_path}")

    json_path = args.json_path

    with open(json_path, "r") as f:
        data = json.load(f)


    # Containers for scores
    score_edtl_orgi_list = []
    score_orgl_edti_list = []
    ratio_list = []

    for item in data:
        score1 = item.get("score_edtl_orgi")
        score2 = item.get("score_orgl_edti")

        if score1 is not None:
            score_edtl_orgi_list.append(score1)
        if score2 is not None:
            score_orgl_edti_list.append(score2)
        if score1 is not None and score2 is not None and score2 != 0:
            ratio_list.append(score1 / score2)

    # Compute means
    mean_score_edtl_orgi = sum(score_edtl_orgi_list) / len(score_edtl_orgi_list) if score_edtl_orgi_list else 0
    mean_score_orgl_edti = sum(score_orgl_edti_list) / len(score_orgl_edti_list) if score_orgl_edti_list else 0
    mean_ratio = sum(ratio_list) / len(ratio_list) if ratio_list else 0

    # Print results
    print(f" Mean score_edtl_orgi: {mean_score_edtl_orgi:.4f}")
    print(f" Mean score_orgl_edti: {mean_score_orgl_edti:.4f}")
    print(f" Mean ratio (score_edtl_orgi / score_orgl_edti): {mean_ratio:.4f}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate hallucination scores.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON file.")
    parser.add_argument("--base_path", type=str, required=True, help="Base path for the masks.")

    args = parser.parse_args()
    main(args)