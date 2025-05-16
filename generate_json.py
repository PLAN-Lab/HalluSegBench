import os
import json
import argparse


def main(args):
    
    # Paths    
    input_json = args.data_ann_path
    output_json = args.output_json_path
    pred_base = args.prdiction_data_path
    
    # Load original JSON
    with open(input_json, "r") as f:
        data = json.load(f)

    # Function to build mask path
    def build_mask_path(subfolder, image_file, ann_id):
        image_id = os.path.splitext(os.path.basename(image_file))[0]
        filename = f"{image_id}_{ann_id}_mask.png"
        return os.path.join(pred_base, subfolder, filename)

    # Add 4 new mask paths for each entry
    for entry in data:
        image_file = entry["image_file"]
        ann_id = entry["ann_id"]

        entry["mask_orgl_orgi"] = build_mask_path("orgl_orgi", image_file, ann_id)
        entry["mask_orgl_edti"] = build_mask_path("orgl_edti", image_file, ann_id)
        entry["mask_edtl_edti"] = build_mask_path("edtl_edti", image_file, ann_id)
        entry["mask_edtl_orgi"] = build_mask_path("edtl_orgi", image_file, ann_id)

    # Save new JSON
    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)

    output_json  # Return path for confirmation


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate JSON with mask paths.")
    parser.add_argument("--data_ann_path", type=str, required=True, help="Path to the original JSON file.")
    parser.add_argument("--output_json_path", type=str, required=True, help="Path to save the new JSON file.")
    parser.add_argument("--prdiction_data_path", type=str, required=True, help="Base path for the masks.")
    args = parser.parse_args()
    main(args)