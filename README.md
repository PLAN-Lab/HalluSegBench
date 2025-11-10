<p align="center">
  <img width="500" src="assets/fig/logo.png" alt="mTSBench Logo"/>
</p>

<h1 align="center">
  <img src="assets/fig/logo_w.png" alt="mTSBench Icon" width="32" style="vertical-align: middle; margin-right: 8px;">
  <b>HalluSegBench: Counterfactual Visual Reasoning for Segmentation Hallucination Evaluation</b>
</h1>


<p align="center">
  <a href="https://huggingface.co/datasets/PLAN-Lab/Hallu">
    <img src="https://img.shields.io/badge/HuggingFace-HalluSegBench-blue?logo=huggingface" alt="Hugging Face badge">
  </a>
</p>

---

Recent progress in vision-language models and segmentation methods has significantly advanced grounded visual understanding. However, these models often exhibit hallucination by producing segmentation masks for objects not grounded in the image content or by incorrectly labeling irrelevant regions. Current evaluation paradigms primarily focus on label or textual hallucinations without manipulating the visual context, limiting their capacity to diagnose critical failures. In response, we introduce HalluSegBench, the first benchmark specifically designed to evaluate hallucination in visual grounding through the lens of counterfactual visual reasoning.
Our benchmark consists of a novel dataset of 1.4K counterfactual image pairs spanning 287 unique object classes, and a set of newly introduced metrics to assess hallucination robustness in reasoning-based segmentation models. Experiments on HalluSegBench with state-of-the-art pixel-grounding models reveal that vision-driven hallucinations are significantly more prevalent than label-driven ones, with models often persisting in false segmentation, highlighting the necessity for counterfactual reasoning to diagnose true visual grounding. We open-source the benchmark at https://huggingface.co/datasets/PLAN-Lab/Hallu to encourage future research in this area.


## Requirements
```bash
pip install -r requirements.txt
```

## Prepare predictions
1. Run your model on all four settings mentioned in the datast and save the masks in four different directories as mentioned below.

The mask names should follow the format of `{image_id}_{ann_id}_mask.png` where `image_id` and `ann_id` is from RefCOCO, also found in `filter_anno.json` in the dataset. 

eg. `COCO_train2014_000000533293_299985_mask.png`

```
├── <path_to_prediction>
│   ├── orgl_orgi  # original label original image
│   ├── orgl_edti  # original label edited image
│   ├── edtl_edti  # edited label edited image
│   ├── edtl_orgi  # edited label original image
```

2. Prepare a json file based on `filter_anno.json` in the dataset
```
python generate_json.py \
    --data_ann_path <path_to_filter_anno.json> \
    --output_json_path <path_to_prediction>/results.json \
    --prdiction_data_path <path_to_prediction> \
```

## Get evaluation scores on the metrics

1. Get consistency based performance metrics
```
python get_consistency.py \
    --json_path <path_to_prediction>/results.json \
    --base_path <path_to_data_base_dir> \
    --pred_base_path <path_to_prediction>
```

2. Get the hallucination based performance metrics
```
python get_hallucination.py \
    --json_path <path_to_prediction>/results.json \
    --base_path <path_to_data_base_dir> \
    --pred_base_path <path_to_prediction>
```
