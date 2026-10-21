# AngioCoT

## Data
- **Dataset:** Pirmoradi, S. (2024). *X-Ray Angiography Images and SYNTAX-Score Dataset*. Figshare. https://doi.org/10.6084/m9.figshare.25801447.v1
- **Publication:** Mahmoudi, S. S., et al. (2025). *X-ray coronary angiogram images and SYNTAX score to develop machine-learning algorithms for CHD diagnosis*. Scientific Data, 12, 471. https://doi.org/10.1038/s41597-025-04727-0
The dataset is not included in this repository. Please obtain it from the original source.


```text
./
├── cot_generation/   # prompt00, multi-round conversation, error correction, convert CoT
├── dataset/          # SFT and DPO data
├── inference/        # SFT model inference
├── evaluation/       # Evaluation metrics
├── requirements.txt
└── README.md
```

Overall pipeline: GT correction, CoT, dataset, SFT, optional DPO, inference, evaluation.

## Setup

Linux / Bash. NVIDIA GPU with bfloat16 support and a CUDA 12.8 compatible driver.

| Component | Experiment version |
|---|---|
| Python | `3.10.19` |
| PyTorch | `2.9.0+cu128` |
| PyTorch CUDA runtime | `12.8` |
| cuDNN | `9.10.2.21` |
| Transformers | `4.57.2` |
| ms-swift | `3.12.3` |
| vLLM | `0.11.2` |

Run commands from the repository root. Use your own folder name and path. Create the environment, install PyTorch for CUDA 12.8, then install the remaining packages:

```bash
conda create -n angiocot python=3.10.19 -y
conda activate angiocot

cd "/path/to/your/angiocot"
python -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
```

## 1. Build CoT

### Paths and model server

Before running a script, replace its `/path/to/your/...` placeholders with your paths.

Image and ground truth (GT) files:

```text
/path/to/your/image_directory/
├── syntax_summary.json
└── patient_001/
    └── *overlay*.png   # Also supports jpg, jpeg and bmp
```

Match each patient folder name to its `Source Folder` value in GT. Replace the example name `patient_001` with your patient folder name.

GT format: a JSON array of patient records.

Edit the settings at the top of `cot_generation/multiauto.py`:

| Setting | Value |
|---|---|
| `DATA_ROOT` | Root folder for patient images |
| `GT_FILENAME` | GT file name; default: `syntax_summary.json` |
| `OUTPUT_ROOT` | Output folder for structured results |
| `SERVER_IP`, `SERVER_PORT` | Model server address; default: `127.0.0.1:8000` |
| `MODEL_PATH` | Model name used by the server |

Start an OpenAI-compatible model server with a `/v1` endpoint. Example using vLLM and a local model:

```bash
MODEL_DIR="/path/to/your/cot_generation_model"
MODEL_NAME="your-served-model-name"

CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_DIR" \
  --served-model-name "$MODEL_NAME" \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image": 24}'
```

Multiple correction rounds may exceed the image limit. Increase `--limit-mm-per-prompt` if this happens.

Keep the server running. In another terminal, activate the `angiocot` environment and enter the repository root. Set `MODEL_PATH` in both generation scripts to the same value as `MODEL_NAME`. For a remote server, replace `127.0.0.1` in `SERVER_IP` and `DEFAULT_BASE_URL` with the server address.

genCoTauto.py: DATA_ROOT_DIR = output folder from `multiauto.py`. Keep the server URL and model name same as `multiauto.py`.

### GT correction and CoT generation

All patients:

```bash
python cot_generation/multiauto.py
```

One patient instead:

```bash
python cot_generation/multiauto.py --patient_folder patient_001
```

Generate CoT:

```bash
python cot_generation/genCoTauto.py
```

After changing `OUTPUT_ROOT`, update `DATA_ROOT_DIR` in the generator and converter.

Output: `OUTPUT_ROOT/<patient>/auto/`, including `step*_final.json` and `cot.json`. Image limit: 24 per patient.

Check that the CoT has ten steps. Fix failed cases or leave them out before conversion.

### Convert JSON for dataset building

`convert_cot.py` paths: DATA_ROOT_DIR for generated CoT, OUTPUT_ROOT for converted CoT. Then run:

```bash
python cot_generation/convert_cot.py
```

Make cot.json ready for the dataset scripts. Nested JSON output: OUTPUT_ROOT/<patient>/cot.json.

It keeps the CoT text, step numbers and line breaks. For missing, empty or invalid CoT fields, it reports an error and skips the patient.

## 2. Build datasets

### Paths

Edit the settings at the top of both `dataset/build_*dataset.py` scripts:

| Setting | Value |
|---|---|
| `PREFIX` | Base folder for relative paths |
| `DATA_ROOT` | Root folder for patient images |
| `COT_PATH` | Root folder for patient CoT files |
| `GT_PATH` | Path to `syntax_summary.json` |
| `SAVED_PATH` | Output prefix, without an extension or `_train` / `_val` suffix |

Use same input paths for both dataset scripts. COT_PATH is the converter output folder. Answer-only also reads the patient list here.

Use full paths to replace the placeholders. For relative paths, the scripts add `PREFIX` before the path. The CoT builder reads `COT_PATH/<patient>/cot.json`.

CoT input format:

```json
{"cot": "{\"cot\": \"1) First step.\\n2) Second step.\"}"}
```

### Build commands

```bash
python dataset/build_sft_dataset.py
python dataset/build_cot_sft_dataset.py
```

Example output prefixes (`SAVED_PATH`):

| Dataset | Output prefix |
|---|---|
| Answer only | `/path/to/your/datasets/answer_sft` |
| CoT + answer | `/path/to/your/datasets/cot_sft` |

Each prefix produces `_train.jsonl` and `_val.jsonl`. Use the same prefix for `DATA_PREFIX` in the training command.

Patient split: about 80% training and 20% validation within each left/right dominance group, with seed 42. SYNTAX classes for SFT and evaluation: `Low <= 22` and `High (Medium included) > 22`.

Both builders use the same oversampling method. After the patient split, they repeat samples from the smaller training dominance group until both groups have the same size. No oversampling in validation.

For clean and masked CoT, use `build_cot_sft_dataset.py` with separate `COT_PATH` and `SAVED_PATH` values. Use the same patient set and split across experiments.

## 3. Train, run inference and evaluate

### SFT training

Models: `Qwen3-VL-4B-Thinking`, `Qwen3-VL-8B-Thinking`, `InternVL2-8B`, `MiniCPM-V-2_6`.

SFT reads the system and user prompts from the JSONL messages. Set `SYSTEM_PROMPT` and `USER_PROMPT` in the dataset builders; rebuild the datasets after changing them.

Set the model, dataset and output paths below. Change `CUDA_VISIBLE_DEVICES` to use your GPU. Adjust `--save_steps`, `--eval_steps`, `--per_device_train_batch_size` and `--max_length` for your hardware and dataset.

```bash
MODEL_DIR="/path/to/your/model"
DATA_PREFIX="/path/to/your/datasets/answer_sft"
OUTPUT_DIR="/path/to/your/training_output"

export PYTHONUNBUFFERED=1
export SWIFT_PATCH_CONV3D=1
mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 swift sft \
  --model "$MODEL_DIR" \
  --train_type lora \
  --dataset "${DATA_PREFIX}_train.jsonl" \
  --val_dataset "${DATA_PREFIX}_val.jsonl" \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --gradient_accumulation_steps 4 \
  --max_length 14672 \
  --gradient_checkpointing true \
  --output_dir "$OUTPUT_DIR" \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs '{"min_lr": 1e-7}' \
  --dataloader_num_workers 8 \
  > "$OUTPUT_DIR/train.log" 2>&1
```

For CoT or masked CoT training, use its `SAVED_PATH` as `DATA_PREFIX` and set a separate `OUTPUT_DIR`.

### SFT inference

Use `inference/sft_infer_vllm.py` for answer-only, clean CoT and masked CoT models. Input: the matching `_val.jsonl`. Default engine: ms-swift 3.12.3 `PtEngine`. Add `--use_vllm` to the inference command to use vLLM.

After LoRA training, merge your chosen checkpoint with the base model. Set `ADAPTER_DIR` to the checkpoint folder and `MERGED_MODEL_DIR` to the merged model output folder:

```bash
ADAPTER_DIR="/path/to/your/checkpoint"
MERGED_MODEL_DIR="/path/to/your/merged_model"

swift export \
  --adapters "$ADAPTER_DIR" \
  --merge_lora true \
  --output_dir "$MERGED_MODEL_DIR"
```

Edit `PREFIX` at the top of the inference script if you use relative paths. You can also use full paths for the three arguments below. For `--ckpt_dir`, use the complete model folder with weights and processor files:

```bash
CUDA_VISIBLE_DEVICES=0 python inference/sft_infer_vllm.py \
  --val_file "/path/to/your/datasets/answer_sft_val.jsonl" \
  --ckpt_dir "$MERGED_MODEL_DIR" \
  --save_dir "/path/to/your/inference_output" \
  --batch_size 16
```

For clean and masked CoT, use the matching validation set and model folder.

Output: `save_dir/inference_results.json`, with raw answers and the parsed `gt_parsed` and `pred_parsed` fields.

### Evaluation

Use `evaluation/eval.py` for all settings. Put the path to save_dir/inference_results.json in INPUT_FILE, then run:

```bash
python evaluation/eval.py
```

Input: a JSON array. Each record includes `gt_parsed` and `pred_parsed` answer dictionaries. Example:

```json
[
  {
    "gt_parsed": {"dominance": "right", "lesion_count": 2},
    "pred_parsed": {"dominance": "right", "lesion_count": 3}
  }
]
```

Classification metrics: Accuracy, macro Precision, Recall and F1, plus a confusion matrix. Regression metrics for `lesion_count` and `syntax_score`: MSE, RMSE and MAE. The evaluator counts missing classification predictions as `parse_error` and uses 0 for missing regression predictions.

Output: `<input_name>_full_metrics.json` in the input file's folder.

## 4. DPO (optional)

### Build preference data

Use the clean CoT `_train.jsonl` and its merged SFT model to generate four candidates per input. Each pair uses the reference CoT + answer as the chosen response and a candidate rejected by the Qwen3-VL-32B judge as the rejected response.

Set these paths at the top of `dataset/generate_dpo_data.py`:

| Setting | Path |
|---|---|
| `MODEL_SFT_PATH` | Merged CoT SFT model |
| `MODEL_JUDGE_PATH` | Qwen3-VL-32B-Instruct model |
| `INTERMEDIATE_FILE` | Candidate cache `.jsonl` |

Create the cache and output parent folders. Image paths in the input JSONL must be accessible from the repository root; use full paths.

```bash
mkdir -p "/path/to/your/datasets"
python dataset/generate_dpo_data.py \
  --input_file "/path/to/your/datasets/cot_sft_train.jsonl" \
  --output_file "/path/to/your/datasets/dpo_train.jsonl"
```

An existing `INTERMEDIATE_FILE` skips generation. Remove it or use a new path when changing the input or SFT model, or after incomplete generation.

Both stages use GPU 0. The 32B judge must fit on that GPU. Adjust `--generator_batch_size`, `--judge_batch_size`, `max_model_len` and GPU memory settings for your hardware.

### DPO training

Use the same merged CoT SFT model and generated DPO JSONL as above. Train a new LoRA adapter; the frozen merged SFT model serves as the reference.

```bash
MODEL_DIR="/path/to/your/merged_cot_sft_model"
DPO_DATA="/path/to/your/datasets/dpo_train.jsonl"
OUTPUT_DIR="/path/to/your/dpo_output"

export PYTHONUNBUFFERED=1
export SWIFT_PATCH_CONV3D=1
mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 MAX_PIXELS=1003520 swift rlhf \
  --rlhf_type dpo \
  --model "$MODEL_DIR" \
  --dataset "$DPO_DATA" \
  --load_from_cache_file true \
  --split_dataset_ratio 0.01 \
  --train_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --learning_rate 5e-6 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --gradient_accumulation_steps 8 \
  --eval_steps 20 \
  --save_steps 20 \
  --logging_steps 4 \
  --max_length 8192 \
  --output_dir "$OUTPUT_DIR" \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --rpo_alpha 0.1 \
  --dataset_num_proc 4 \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs '{"min_lr": 1e-8}' \
  > "$OUTPUT_DIR/train.log" 2>&1
```

This command reserves about 1% of the preference training data for validation. `--rpo_alpha 0.1` adds a chosen-response negative log-likelihood term weighted by 0.1 to the DPO loss. Adjust batch sizes, `--save_steps`, `--eval_steps` and `--max_length` for your hardware and data size.

Reuse the export, inference and evaluation steps above with the DPO checkpoint and the original patient-level clean CoT validation set. Use a separate merged-model output folder for DPO.
