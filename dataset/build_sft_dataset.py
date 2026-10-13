"""Build patient-level answer-only SFT data with a binary SYNTAX endpoint."""

import glob
import os
import re
import json
import random
from typing import List, Dict


# Configuration
# Replace the path placeholders before running.
PREFIX = "/path/to/your/project"
DATA_ROOT = "/path/to/your/image_directory"
COT_PATH = "/path/to/your/cot_directory"
GT_PATH = "/path/to/your/syntax_summary.json"
SAVED_PATH = "/path/to/your/datasets/answer_sft"
MAX_IMAGES = 24

SYSTEM_PROMPT = r"""
### Role
You are a medical assistant trained in coronary angiography interpretation.

### Task
Your task is to analyze coronary angiography images of the current patient and output ONE JSON object with EXACTLY the following keys:

### Key Attributes
- dominance: A string representing the coronary dominance type.
    - Allowed values: "left", "right".
    - "left": PDA originates from the left circumflex artery (LCX).
    - "right": Posterior descending artery (PDA) originates from the right coronary artery (RCA).

- lesion_count: The number of significant lesions identified in the angiography.

- syntax_classification: The SYNTAX complexity class based on the estimated syntax score (An estimated numerical SYNTAX score for this patient (non-negative integer)).
    * "Low" for scores belong to [0, 22]
    * "High" for scores > 22.

- lesions greater than 20mm in length: true/false
- lesions with total occlusion: true/false
- lesions with thrombus: true/false
- lesions with bifurcation: true/false
- lesions with heavy calcification: true/false

### IMPORTANT OUTPUT RULES:
- **Structure:** Your final output must be strictly enclosed within `<answer>` tags.
- **JSON Constraint (Crucial):**
  - The content strictly inside the `<answer>` tags must be a raw JSON object.
  - **The first non-whitespace character inside `<answer>` MUST be `{`.**
  - **Do NOT** include markdown, bullet points, code fences (like ```json) or any introductory text inside the `<answer>` tags.

### IMAGE INTERPRETATION:
- Purple segments: Coronary vessel skeleton.
- Red dots: Branching points.
- Blue dots: Midpoints of blood vessels.
- Green outlines with yellow backgrounds: Segmentation mask regions (the yellow area inside the green outline is the vessel mask, which could be used to recognize the narrowings).
"""

USER_PROMPT = r"""
Analyze the {NUM_IMAGES} given coronary angiography images of the current patient and output a JSON object with the following keys:

- dominance
- lesion_count
- syntax_classification
- Does the patient have any lesions greater than 20mm in length?
- Does the patient have any lesions with total occlusion?
- Does the patient have any lesions with thrombus?
- Does the patient have any lesions with bifurcation?
- Does the patient have any lesions with heavy calcification?
"""


def get_patient_ids(path: str):
    patient_ids = [
        item for item in os.listdir(path)
        if os.path.isdir(os.path.join(path, item)) and not item.startswith('.')
    ]
    return sorted(patient_ids, key=_natural_sort_key)


# Image utilities
def _natural_sort_key(path: str):
    name = os.path.basename(path)
    parts = re.split(r"(\d+)", name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def collect_overlay_images(folder: str, max_imgs: int) -> List[str]:
    patterns = [
        "*overlay*.jpg", "*overlay*.jpeg", "*overlay*.png", "*overlay*.bmp",
        "*overlay*.JPG", "*overlay*.JPEG", "*overlay*.PNG", "*overlay*.BMP",
    ]
    images: List[str] = []
    for pat in patterns:
        images.extend(glob.glob(os.path.join(folder, pat)))
    images = sorted(set(images), key=_natural_sort_key)
    return images[:max_imgs]


# Answers utilities
def get_gt(gt_path: str):
    with open(gt_path, 'r', encoding='utf-8') as gt_file:
        gt_data = json.load(gt_file)
        gt_keys = next(k for k in gt_data[0] if "Source Folder" in k)
        gt_mappings = {item[gt_keys]: item for item in gt_data}
    return gt_mappings


def get_any_lessions(cur_gt: Dict):
    try:
        lession_nums = int(cur_gt.get("Number of lesions", 0))
    except ValueError:
        lession_nums = 0

    target_attributes = [
        "Length >20 mm",
        "Total occlusion",
        "Thrombus",
        "Bifurcation",
        "Heavy calcification"
    ]
    lession_res = {}

    for attr in target_attributes:
        lession_res[attr] = any(
            str(cur_gt.get(f"Lesion {i}-{attr}", "")).strip().lower().startswith('y')
            for i in range(1, lession_nums + 1)
        )

    return lession_res


def build_answers(gt_mappings: Dict, patient_id: str):
    cur_gt = gt_mappings[patient_id]
    cur_answer = {}

    lession_res = get_any_lessions(cur_gt)
    dominance_val = cur_gt.get("Dominance", "unknown")
    cur_answer["dominance"] = dominance_val if dominance_val else "unknown"
    num_lesions_val = cur_gt.get("Number of lesions", 0)
    cur_answer["lesion_count"] = int(num_lesions_val) if num_lesions_val else 0
    try:
        raw_score = cur_gt.get("SYNTAX Score", "0")
        score_val = float(raw_score)
    except ValueError:
        score_val = 0.0

    # High merges Intermediate and High.
    if score_val <= 22:
        cur_answer["syntax_classification"] = "Low"
    else:
        cur_answer["syntax_classification"] = "High"

    cur_answer["lesions greater than 20mm in length"] = lession_res["Length >20 mm"]
    cur_answer["lesions with total occlusion"] = lession_res["Total occlusion"]
    cur_answer["lesions with thrombus"] = lession_res["Thrombus"]
    cur_answer["lesions with bifurcation"] = lession_res["Bifurcation"]
    cur_answer["lesions with heavy calcification"] = lession_res["Heavy calcification"]

    return json.dumps(cur_answer, ensure_ascii=False)


# Messages utilities
def build_messages(gt_mappings: Dict, patient_id: str):
    cur_answer = build_answers(gt_mappings, patient_id)
    response = f"<answer>\n{cur_answer}\n</answer>"

    system_prompt = {"role": "system", "content": SYSTEM_PROMPT}
    user_prompt = {"role": "user", "content": USER_PROMPT}
    assistant_prompt = {"role": "assistant", "content": response}
    return [system_prompt, user_prompt, assistant_prompt]


# Build datasets
def main():
    random.seed(42)

    gt_path = os.path.join(PREFIX, GT_PATH)
    gt_mappings = get_gt(gt_path)

    base_saved_path = os.path.join(PREFIX, SAVED_PATH)

    cot_path = os.path.join(PREFIX, COT_PATH)
    patient_ids = get_patient_ids(cot_path)

    samples_by_dominance = {
        "left": [],
        "right": [],
    }

    print(f"Processing {len(patient_ids)} patients.")

    for patient_id in patient_ids:
        cur_gt = gt_mappings.get(patient_id, {})
        dominance = str(cur_gt.get("Dominance", "unknown")).strip().lower()

        if "left" in dominance:
            group_key = "left"
        elif "right" in dominance:
            group_key = "right"
        else:
            print(f"Warning: Unknown dominance for {patient_id}; skipped.")
            continue

        imgs_folder = os.path.join(PREFIX, DATA_ROOT, patient_id)
        imgs = collect_overlay_images(folder=imgs_folder, max_imgs=MAX_IMAGES)

        if not imgs:
            print(f"Warning: No images for {patient_id}; skipped.")
            continue

        num_imgs = len(imgs)
        image_tags = "<image>" * num_imgs

        messages = build_messages(gt_mappings, patient_id)
        messages[1]["content"] = messages[1]["content"].format(
            NUM_IMAGES=num_imgs)
        messages[1]["content"] = image_tags + messages[1]["content"]

        patient_dict = {"images": imgs, "messages": messages}

        samples_by_dominance[group_key].append(patient_dict)

    train_groups = {}
    val_samples = []

    print("\nPatient split:")
    for key, samples in samples_by_dominance.items():
        random.shuffle(samples)

        total_count = len(samples)
        if total_count == 0:
            continue

        split_idx = int(total_count * 0.8)

        if split_idx == 0 and total_count > 0:
            split_idx = total_count

        group_train = samples[:split_idx]
        group_val = samples[split_idx:]

        train_groups[key] = group_train
        val_samples.extend(group_val)

        print(
            f"  {key}: Total {total_count}, Train {len(group_train)}, Val {len(group_val)}")

    train_samples = []
    max_train_count = max((len(group) for group in train_groups.values()), default=0)
    print("\nBalancing training by dominance.")
    for key, group in train_groups.items():
        count = len(group)
        if count < max_train_count:
            q, r = divmod(max_train_count, count)
            train_samples.extend(group * q + group[:r])
            print(f"  Upsampled '{key}': {count} -> {max_train_count}")
        else:
            train_samples.extend(group)

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    train_path = f"{base_saved_path}_train.jsonl"
    val_path = f"{base_saved_path}_val.jsonl"
    os.makedirs(os.path.dirname(base_saved_path), exist_ok=True)

    print(f"\nWriting {len(train_samples)} samples to {train_path}")
    with open(train_path, 'w', encoding='utf-8') as f_train:
        for item in train_samples:
            f_train.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Writing {len(val_samples)} samples to {val_path}")
    with open(val_path, 'w', encoding='utf-8') as f_val:
        for item in val_samples:
            f_val.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
