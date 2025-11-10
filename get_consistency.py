import os
import cv2
import json
import argparse

import numpy as np


def compute_iou(mask1, mask2, image_shape):
    mask1 = cv2.resize(mask1.astype(np.uint8), (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    mask2 = cv2.resize(mask2.astype(np.uint8), (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)

    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    return intersection / union if union > 0 else 0.0

def main(args):

    json_path = args.json_path
    base_path = args.base_path
    pred_base_path = args.pred_base_path
    with open(json_path, "r") as f:
        data = json.load(f)
    
    ious_1, ious_2, ious_3 = [], [], []
    diff_12, diff_13 = [], []


    for item in data:
        try:
            image_ann = f"{item['factual_image_path']} | ann_id: {item['ann_id']}"
            print(f"\n {image_ann}")

            gt1_path = os.path.join(base_path, item["factual_mask_path"])
            gt3_path = os.path.join(base_path, item["counterfactual_mask_path"])
            pred1_path = os.path.join(pred_base_path, item.get("mask_orgl_orgi"))
            pred2_path = os.path.join(pred_base_path, item.get("mask_edtl_orgi"))
            pred3_path = os.path.join(pred_base_path, item.get("mask_orgl_edti"))

            if not (os.path.exists(gt1_path) and os.path.exists(gt3_path)):
                print("Ground truth mask not found, skipped.")
                continue

            gt1 = cv2.imread(gt1_path, 0) > 0
            gt3 = cv2.imread(gt3_path, 0) > 0
            image_shape = gt1.shape

            iou1 = iou2 = iou3 = None

            # Group 1
            if pred1_path and os.path.exists(pred1_path):
                pred1 = cv2.imread(pred1_path, 0) > 0
                iou1 = compute_iou(gt1, pred1, image_shape)
                ious_1.append(iou1)
            else:
                print("⚠️ mask_orgl_origi missing")

            # Group 2
            if pred2_path and os.path.exists(pred2_path):
                pred2 = cv2.imread(pred2_path, 0) > 0
                iou2 = compute_iou(gt1, pred2, image_shape)
                ious_2.append(iou2)
            else:
                print("⚠️ mask_edtl_edti missing")

            # Group 3
            if pred3_path and os.path.exists(pred3_path):
                pred3 = cv2.imread(pred3_path, 0) > 0
                iou3 = compute_iou(gt3, pred3, image_shape)
                ious_3.append(iou3)
            else:
                print("⚠️ mask_orgl_edti missing")

            # Diffs
            if iou1 is not None and iou2 is not None:
                diff = (iou1 - iou2)
                diff_12.append(diff)
            if iou1 is not None and iou3 is not None:
                diff = (iou1 - iou3)
                diff_13.append(diff)

        except Exception as e:
            print(f" Error processing {item.get('factual_image_path')}: {e}")


    print("\n IOU Summary:")
    print(f"1. IOU(mask_file, mask_orgl_orgi): mean = {np.mean(ious_1):.4f}")
    print(f"2. IOU(mask_file, mask_edtl_orgi): mean = {np.mean(ious_2):.4f}")
    print(f"3. IOU(new_mask_path, mask_orgl_edti): mean = {np.mean(ious_3):.4f}")

    print("\n IOU Difference Summary:")
    print(f"(1 - 2): mean = {np.mean(diff_12):.4f}")
    print(f"(1 - 3): mean = {np.mean(diff_13):.4f}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute IOU and differences between masks.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON file.")
    parser.add_argument("--base_path", type=str, required=True, help="Base path for the masks.")
    parser.add_argument("--pred_base_path", type=str, required=True, help="Base path for predicted masks if needed.")
    
    args = parser.parse_args()
    
    main(args)