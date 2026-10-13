"""GT-guided structured states."""

import argparse
import os
import re
import glob
import json
import base64
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# Config
# Set paths.
DATA_ROOT = "/path/to/your/image_directory"
PATIENT_FOLDER = "patient_001"  # CLI override: --patient_folder
MAX_IMAGES = 24

SERVER_IP = "127.0.0.1"
SERVER_PORT = 8000
BASE_URL = f"http://{SERVER_IP}:{SERVER_PORT}/v1"
MODEL_PATH = "your-served-model-name"

OPENAI_API_KEY = "not-used"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_FREQUENCY_PENALTY = 0.0

MAX_TOKENS_STEP00 = 4000
MAX_TOKENS_JSON_TASK = 2500
MAX_TOKENS_REPAIR = 1200

MAX_HISTORY_MESSAGES = 20

OUTPUT_ROOT = "/path/to/your/structured_cot_directory"
PROMPT_FILE_STEP00 = "step_00_image_attribution.txt"
PROMPT_DIR = os.path.dirname(os.path.abspath(__file__))

GT_FILENAME = "syntax_summary.json"

TRY_RESPONSE_FORMAT_JSON = True
TRY_VLLM_GUIDED_JSON = True

# Patient turn limits
MAX_TURNS_PER_PATIENT = 50
WARN_TURN_AT = 40


# Step prompts
STEP01_TEMPLATE = r"""You are a medical assistant trained in coronary angiography interpretation.

TASK
Determine coronary dominance for the CURRENT patient from the provided {NUM_IMAGES} angiography overlay images.
Dominance is defined by which artery gives rise to the posterior descending artery (PDA) and supplies the inferior interventricular septum.
You may use prior-step context if available: {CONTEXT_JSON}

IMAGE INTERPRETATION KEY
- Purple segments: Coronary vessel skeleton.
- Red dots: Branching points.
- Blue dots: Midpoints of blood vessels.
- Green outlines with yellow backgrounds: Segmentation mask regions.

GUIDANCE (use internally)
- Right-dominant: PDA arises from RCA.
- Left-dominant: PDA arises from LCx (rarely LAD continuation).
- Co-dominant: mixed supply or uncertain PDA origin with meaningful LCx posterolateral contribution.
- Use multiple projections; if PDA/crux is not well seen, keep confidence moderate and state uncertainty.

OUTPUT FORMAT
Return ONE JSON object with EXACTLY these keys:
- dominance_label: one of ["right", "left", "co", "unclear"]
- right_dominance_confidence: a float in [0.0, 1.0] representing confidence that the system is RIGHT-dominant
- pda_source: one of ["RCA", "LCx", "LAD", "unclear"]
- evidence_by_image: a list of EXACTLY {NUM_IMAGES} strings, one per image in the SAME ORDER, briefly stating what that image shows relevant to dominance (or "not informative for dominance").

STRICT OUTPUT RULES
- Output ONLY the JSON object (no markdown, no code fences, no extra text).
- The first non-whitespace character MUST be '{'.
- Do not add any other top-level keys.
"""

STEP02_TEMPLATE = r"""You are a medical assistant trained in coronary angiography interpretation.

TASK
Create a lesion inventory (where are the clinically relevant stenoses) for the CURRENT patient using {NUM_IMAGES} angiography overlay images.
You may use prior-step context if available: {CONTEXT_JSON}

IMPORTANT CLINICAL NOTES (use internally)
- Review images in multiple projections; overlap can mimic a stenosis (pseudo-lesion). Prefer lesions seen consistently across views.
- If diffuse atherosclerosis makes boundaries unclear, apply a spacing rule: lesions <3 vessel diameters apart should be treated as ONE lesion rather than multiple separate lesions.
- Focus on lesions that plausibly reach ≥50% diameter stenosis in vessels that appear ≥1.5 mm caliber. If uncertain, still list but mark uncertainty.

IMAGE INTERPRETATION KEY
- Purple segments: Coronary vessel skeleton.
- Red dots: Branching points.
- Blue dots: Midpoints of blood vessels.
- Green outlines with yellow backgrounds: Segmentation mask regions.

OUTPUT FORMAT
Return ONE JSON object with EXACTLY these keys:
- lesion_count_est: non-negative integer
- lesions: a list of lesion objects (length can be 0+)

Each lesion object MUST have EXACTLY these keys:
- lesion_id: string like "L1", "L2", ...
- vessel_guess: one of ["LM", "LAD", "LCx", "RCA", "Branch", "Unknown"]
- segment_numbers: a list of strings (e.g., ["3"] or ["11","12a"]); empty list if unknown
- suspected_significance: one of ["likely>=50%", "likely<50%", "uncertain"]
- evidence_images: a list of integers (0-based image indices) that support this lesion
- notes_short: short text (<= 25 words) describing the lesion location impression

STRICT OUTPUT RULES
- Output ONLY the JSON object (no markdown, no code fences, no extra text).
- The first non-whitespace character MUST be '{'.
- Do not add any other top-level keys.
"""

STEP025_TEMPLATE = STEP02_TEMPLATE

STEP026_TEMPLATE = r"""You are a medical assistant trained in coronary angiography interpretation.

TASK
Assess coronary calcification from the {NUM_IMAGES} angiography overlay images.
You MUST align any "heavy calcification" determination to the lesions in the inventory context.
Do NOT introduce new lesions.

CONTEXT (must follow)
{CONTEXT_JSON}

CALCIFICATION GUIDANCE (use internally)
- Look for radiopaque densities along the vessel wall that persist across frames/projections.
- Grade overall calcification as none / mild / moderate / heavy. If uncertain, say "uncertain".
- "Heavy calcification" for SYNTAX purposes should be reserved for clearly substantial calcific burden affecting lesion handling; if not clearly heavy, choose "No" or "Unclear" rather than "Yes".

IMAGE INTERPRETATION KEY
- Purple segments: Coronary vessel skeleton.
- Red dots: Branching points.
- Blue dots: Midpoints of blood vessels.
- Green outlines with yellow backgrounds: Segmentation mask regions.

OUTPUT FORMAT
Return ONE JSON object with EXACTLY these keys:
- overall_calcification_grade: one of ["none", "mild", "moderate", "heavy", "uncertain"]
- calcification_by_vessel: a list (0+)
- heavy_calcification_by_lesion: a list with one entry per lesion_id from the inventory

Each calcification_by_vessel entry MUST have EXACTLY these keys:
- vessel: one of ["LM", "LAD", "LCx", "RCA", "Branch", "Unknown"]
- grade: one of ["none", "mild", "moderate", "heavy", "uncertain"]
- segments_involved: a list of strings (e.g., ["11","12a"]) or empty list
- evidence_images: a list of integers (0-based image indices)

Each heavy_calcification_by_lesion entry MUST have EXACTLY these keys:
- lesion_id: string matching the inventory
- heavy_calcification: one of ["Yes", "No", "Unclear"]
- evidence_images: a list of integers (0-based image indices) or empty list

STRICT OUTPUT RULES
- Output ONLY the JSON object (no markdown, no code fences, no extra text).
- The first non-whitespace character MUST be '{'.
- Do not add any other top-level keys.
"""

STEP03_TEMPLATE = r"""You are a medical assistant trained in coronary angiography interpretation.

TASK
Extract SYNTAX-relevant lesion attributes for each lesion in the existing inventory.
Do NOT introduce new lesions.
If a feature cannot be determined confidently from angiography overlays, use "Unclear".

CONTEXT (must follow)
{CONTEXT_JSON}

SYNTAX-ATTRIBUTE DEFINITIONS (use internally)
- Total occlusion: no antegrade flow (TIMI 0) through the segment.
- Occlusion adverse features: occlusion age ≥3 months, blunt stump, bridging collaterals, side branch at occlusion, first segment beyond occlusion visualized.
- Trifurcation and bifurcation: specify presence, and if bifurcation present provide Medina classification if possible.
- Bifurcation angle: consider >70° as adverse if clearly wide.
- Aorto-ostial: lesion within ~3 mm of the vessel origin.
- Severe tortuosity: marked bends proximal to lesion that would complicate PCI.
- Thrombus: intraluminal filling defect or haziness consistent with thrombus.

IMAGE INTERPRETATION KEY
- Purple segments: Coronary vessel skeleton.
- Red dots: Branching points.
- Blue dots: Midpoints of blood vessels.
- Green outlines with yellow backgrounds: Segmentation mask regions.

OUTPUT FORMAT
Return ONE JSON object with EXACTLY this key:
- lesion_attributes: a list with one object per lesion_id from the inventory (same ordering as the inventory if available)

Each lesion attribute object MUST have EXACTLY these keys:
- lesion_id: string
- total_occlusion: one of ["Yes", "No", "Unclear"]
- occlusion_features: an object with EXACTLY these keys:
  - occlusion_age_ge_3_months: one of ["Yes", "No", "Unclear"]
  - blunt_stump: one of ["Yes", "No", "Unclear"]
  - bridging_collaterals: one of ["Yes", "No", "Unclear"]
  - side_branch_at_occlusion: one of ["Yes", "No", "Unclear"]
  - first_segment_beyond_occlusion_visualized: one of ["Yes", "No", "Unclear"]
- trifurcation: one of ["Yes", "No", "Unclear"]
- bifurcation: one of ["Yes", "No", "Unclear"]
- medina: string (e.g., "1,1,0") or "" if not applicable/unclear
- bifurcation_angulation_gt_70: one of ["Yes", "No", "Unclear"]
- aorto_ostial: one of ["Yes", "No", "Unclear"]
- severe_tortuosity: one of ["Yes", "No", "Unclear"]
- thrombus: one of ["Yes", "No", "Unclear"]
- evidence_images: a list of integers (0-based image indices) or empty list

STRICT OUTPUT RULES
- Output ONLY the JSON object (no markdown, no code fences, no extra text).
- The first non-whitespace character MUST be '{'.
- Do not add any other top-level keys.
"""

STEP04_TEMPLATE = r"""You are a medical assistant trained in coronary angiography interpretation.

TASK
Estimate the SYNTAX score for the current patient using ONLY the structured results provided in the context.
Do NOT infer additional lesions or attributes beyond what is explicitly present in the context.

CONTEXT (must follow)
{CONTEXT_JSON}

SYNTAX SCORING PRINCIPLES (use internally)
- SYNTAX score sums points for each coronary lesion with ≥50% stenosis in vessels ≥1.5 mm.
- Lesion segment location contributes base weight; complexity features add points (total occlusion and adverse occlusion features; trifurcation/bifurcation with Medina and angle; aorto-ostial; severe tortuosity; lesion length >20 mm; heavy calcification; thrombus; diffuse disease/small vessels).
- If lesions are <3 vessel diameters apart they count as one lesion to avoid double counting.
- Provide an estimated integer score and complexity class:
  - Low: 0-22
  - Intermediate: 23-32
  - High: >=33

OUTPUT FORMAT
Return ONE JSON object with EXACTLY these keys:
- syntax_score_est: non-negative integer
- syntax_class: one of ["Low", "Intermediate", "High"]
- score_rationale_short: a short explanation (<= 60 words) referencing only the context (e.g., number of significant lesions, presence/absence of bifurcation, CTO, length>20mm, heavy calcification, diffuse disease).

STRICT OUTPUT RULES
- Output ONLY the JSON object (no markdown, no code fences, no extra text).
- The first non-whitespace character MUST be '{'.
- Do not add any other top-level keys.
"""


# Utilities
def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def save_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_json(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_gt_path() -> str:
    candidates = [
        os.path.join(DATA_ROOT, GT_FILENAME),
        GT_FILENAME,
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(os.path.join(DATA_ROOT, GT_FILENAME))


# Images
def _natural_sort_key(path: str):
    name = os.path.basename(path)
    parts = re.split(r"(\d+)", name)
    key = []
    for p in parts:
        key.append(int(p) if p.isdigit() else p.lower())
    return key


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


def encode_image_to_data_url(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".png":
        mime = "image/png"
    elif ext == ".bmp":
        mime = "image/bmp"
    else:
        mime = "image/jpeg"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# Prompts
def render_prompt(template: str, num_images: int, context: Dict[str, Any]) -> str:
    text = template.replace("{NUM_IMAGES}", str(num_images))
    if "{CONTEXT_JSON}" in text:
        text = text.replace("{CONTEXT_JSON}", json.dumps(
            context, ensure_ascii=False, indent=2))
    return text


# Messages
def build_system_message(text: str) -> Dict[str, Any]:
    return {"role": "system", "content": text}


def build_user_message_with_images(text: str, image_paths: List[str]) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for p in image_paths:
        url = encode_image_to_data_url(p)
        content.append({"type": "image_url", "image_url": {"url": url}})
    return {"role": "user", "content": content}


def build_user_text_message(text: str) -> Dict[str, Any]:
    return {"role": "user", "content": text}


def build_assistant_message(text: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": text}


# JSON schemas
def schema_step00() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {"per_image_primary_vessel": {"type": "array"}},
        "required": ["per_image_primary_vessel"],
        "additionalProperties": False,
    }


def schema_step01(num_images: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "dominance_label": {"type": "string"},
            "right_dominance_confidence": {"type": "number"},
            "pda_source": {"type": "string"},
            "evidence_by_image": {"type": "array"},
        },
        "required": ["dominance_label", "right_dominance_confidence", "pda_source", "evidence_by_image"],
        "additionalProperties": False,
    }


def schema_step02_inventory() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {"lesion_count_est": {"type": "integer"}, "lesions": {"type": "array"}},
        "required": ["lesion_count_est", "lesions"],
        "additionalProperties": False,
    }


def schema_step026_calcification() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "overall_calcification_grade": {"type": "string"},
            "calcification_by_vessel": {"type": "array"},
            "heavy_calcification_by_lesion": {"type": "array"},
        },
        "required": ["overall_calcification_grade", "calcification_by_vessel", "heavy_calcification_by_lesion"],
        "additionalProperties": False,
    }


def schema_step03_attributes() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {"lesion_attributes": {"type": "array"}},
        "required": ["lesion_attributes"],
        "additionalProperties": False,
    }


def schema_step04_syntax_score() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "syntax_score_est": {"type": "integer"},
            "syntax_class": {"type": "string"},
            "score_rationale_short": {"type": "string"},
        },
        "required": ["syntax_score_est", "syntax_class", "score_rationale_short"],
        "additionalProperties": False,
    }


def schema_for_step(step_name: str, num_images: int) -> Optional[Dict[str, Any]]:
    if step_name == "step00":
        return schema_step00()
    if step_name == "step01":
        return schema_step01(num_images)
    if step_name in ("step02", "step025"):
        return schema_step02_inventory()
    if step_name == "step026":
        return schema_step026_calcification()
    if step_name == "step03":
        return schema_step03_attributes()
    if step_name == "step04":
        return schema_step04_syntax_score()
    return None


# Model calls
def _trim_history(messages: List[Dict[str, Any]], keep: int = MAX_HISTORY_MESSAGES) -> List[Dict[str, Any]]:
    if len(messages) <= keep:
        return messages
    if messages and messages[0].get("role") == "system":
        return [messages[0]] + messages[-(keep - 1):]
    return messages[-keep:]


def chat_once(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    schema: Optional[Dict[str, Any]] = None,
) -> str:
    messages = _trim_history(messages)

    base_kwargs = dict(
        model=MODEL_PATH,
        messages=messages,
        max_tokens=max_tokens,
        temperature=DEFAULT_TEMPERATURE,
        frequency_penalty=DEFAULT_FREQUENCY_PENALTY,
    )

    if TRY_VLLM_GUIDED_JSON and schema is not None:
        try:
            resp = client.chat.completions.create(
                **base_kwargs,
                extra_body={"guided_json": schema},
            )
            content = resp.choices[0].message.content
            return content if isinstance(content, str) else ("" if content is None else str(content))
        except Exception:
            pass

    if TRY_RESPONSE_FORMAT_JSON and schema is not None:
        try:
            resp = client.chat.completions.create(
                **base_kwargs,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            return content if isinstance(content, str) else ("" if content is None else str(content))
        except Exception:
            pass

    resp = client.chat.completions.create(**base_kwargs)
    content = resp.choices[0].message.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content)


# JSON parsing and repair
def extract_first_json_object(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    start = s.find("{")
    if start < 0:
        return ""

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    return ""


def looks_like_runaway(text: str) -> bool:
    if not text:
        return False
    if len(text) > 8000 and "{" not in text:
        return True
    needle = "the image 2 has a branch"
    if text.lower().count(needle) >= 10:
        return True
    return False


def force_json_only_once(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    reason: str,
    schema: Optional[Dict[str, Any]],
) -> str:
    prompt = (
        "STRICT FIX REQUIRED.\n"
        f"Problem: {reason}\n\n"
        "Rewrite your answer as ONE valid JSON object ONLY.\n"
        "No markdown.\n"
        "No code fences.\n"
        "No extra text.\n"
        "The first non-whitespace character MUST be '{'.\n"
        "If you cannot comply, output an empty JSON object: {}\n"
    )
    messages.append(build_user_text_message(prompt))
    raw = chat_once(client, messages,
                    max_tokens=MAX_TOKENS_REPAIR, schema=schema)
    messages.append(build_assistant_message(raw))
    return raw


def parse_json_strict(
    raw_text: str,
    client: OpenAI,
    messages: List[Dict[str, Any]],
    schema: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    raw_json = extract_first_json_object(raw_text)
    if not raw_json or looks_like_runaway(raw_text):
        reason = "No JSON object found in output." if not raw_json else "Runaway/repetition detected."
        raw_fix = force_json_only_once(client, messages, reason, schema)
        raw_json = extract_first_json_object(raw_fix)

    if not raw_json:
        return {}

    try:
        obj = json.loads(raw_json)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        raw_fix = force_json_only_once(
            client, messages, "JSON parse error.", schema)
        raw_json2 = extract_first_json_object(raw_fix)
        if not raw_json2:
            return {}
        try:
            obj2 = json.loads(raw_json2)
            return obj2 if isinstance(obj2, dict) else {}
        except Exception:
            return {}


# GT fields
def normalize_str(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _lower(x: Any) -> str:
    return normalize_str(x).lower()


def _parse_yes_no_unclear(v: str) -> str:
    s = _lower(v)
    if not s:
        return "Unclear"
    if s in ("yes", "y", "true", "1"):
        return "Yes"
    if s in ("no", "n", "false", "0"):
        return "No"
    if s.startswith("yes"):
        return "Yes"
    if s.startswith("no"):
        return "No"
    return "Unclear"


def _parse_bifurcation_yes_no(v: str) -> Tuple[str, str]:
    s = normalize_str(v)
    if not s:
        return "Unclear", ""
    sl = s.lower()
    if sl.startswith("yes"):
        med = ""
        m = re.search(r"medina\s*([0-2]\s*,\s*[0-2]\s*,\s*[0-2])", sl)
        if m:
            med = m.group(1).replace(" ", "")
        return "Yes", med
    if sl.startswith("no"):
        return "No", ""
    return "Unclear", ""


def load_gt_rows(gt_path: str) -> List[Dict[str, Any]]:
    data = load_json(gt_path)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for k in ("rows", "data", "cases", "patients"):
            if k in data and isinstance(data[k], list):
                return [x for x in data[k] if isinstance(x, dict)]
    raise RuntimeError(
        "GT JSON format not recognized: expected a list of row-objects.")


def get_patient_row(rows: List[Dict[str, Any]], patient_folder: str) -> Dict[str, Any]:
    target = patient_folder.strip()
    for row in rows:
        sf_val = None
        for k, v in row.items():
            if isinstance(k, str) and "source folder" in k.lower():
                sf_val = normalize_str(v)
                break
        if sf_val == target:
            return row
    raise RuntimeError(
        f"Cannot find patient '{patient_folder}' by 'Source Folder' in syntax_summary.json")


def _parse_segment_list(seg_str: Any) -> List[str]:
    s = normalize_str(seg_str)
    if not s:
        return []
    parts = re.split(r"[,\s;/]+", s)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out


def gt_step00_from_row(_: Dict[str, Any]) -> Dict[str, Any]:
    return {"per_image_primary_vessel": []}


def gt_step01_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    dom = _lower(row.get("Dominance", ""))
    if dom in ("codominant", "co-dominant"):
        dom = "co"
    if dom not in ("right", "left", "co", "unclear"):
        dom = "unclear"
    pda = {"right": "RCA", "left": "LCx", "co": "unclear",
           "unclear": "unclear"}.get(dom, "unclear")
    return {"dominance_label": dom, "pda_source": pda}


def gt_step02_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    n_str = normalize_str(row.get("Number of lesions", "0"))
    try:
        n = int(float(n_str))
    except Exception:
        n = 0

    lesions = []
    for i in range(1, n + 1):
        seg_key = f"Lesion {i}-Segment numbers involved"
        segs = _parse_segment_list(row.get(seg_key, ""))

        lesions.append(
            {
                "lesion_id": f"L{i}",
                "vessel_guess": "Unknown",
                "segment_numbers": segs,
                "suspected_significance": "uncertain",
                "evidence_images": [],
                "notes_short": "",
            }
        )
    return {"lesion_count_est": n, "lesions": lesions}


def gt_step026_from_row(row: Dict[str, Any], lesion_count: int) -> Dict[str, Any]:
    heavy_list = []
    for i in range(1, lesion_count + 1):
        hk = f"Lesion {i}-Heavy calcification"
        heavy = _parse_yes_no_unclear(normalize_str(row.get(hk, "")))
        heavy_list.append(
            {"lesion_id": f"L{i}", "heavy_calcification": heavy, "evidence_images": []})

    return {
        "overall_calcification_grade": "uncertain",
        "calcification_by_vessel": [],
        "heavy_calcification_by_lesion": heavy_list,
    }


def gt_step03_from_row(row: Dict[str, Any], lesion_count: int) -> Dict[str, Any]:
    attrs = []
    for i in range(1, lesion_count + 1):
        to_k = f"Lesion {i}-Total occlusion"
        m3_k = f"Lesion {i}-More than 3 months"
        blunt_k = f"Lesion {i}-Blunt stump"
        bridge_k = f"Lesion {i}-Bridging"
        side_k = f"Lesion {i}-Side branch"
        beyond_k = f"Lesion {i}-The first segment beyond the T.O. visualized by contrast:"
        tri_k = f"Lesion {i}-Trifurcation"
        bif_k = f"Lesion {i}-Bifurcation"
        ang_k = f"Lesion {i}-Bifurcation angulation"
        ost_k = f"Lesion {i}-Aorto Ostial lesion"
        tort_k = f"Lesion {i}-Severe Tortuosity"
        thr_k = f"Lesion {i}-Thrombus"

        bif, med = _parse_bifurcation_yes_no(normalize_str(row.get(bif_k, "")))

        attrs.append(
            {
                "lesion_id": f"L{i}",
                "total_occlusion": _parse_yes_no_unclear(normalize_str(row.get(to_k, ""))),
                "occlusion_features": {
                    "occlusion_age_ge_3_months": _parse_yes_no_unclear(normalize_str(row.get(m3_k, ""))),
                    "blunt_stump": _parse_yes_no_unclear(normalize_str(row.get(blunt_k, ""))),
                    "bridging_collaterals": _parse_yes_no_unclear(normalize_str(row.get(bridge_k, ""))),
                    "side_branch_at_occlusion": _parse_yes_no_unclear(normalize_str(row.get(side_k, ""))),
                    "first_segment_beyond_occlusion_visualized": "Yes"
                    if normalize_str(row.get(beyond_k, ""))
                    else "Unclear",
                },
                "trifurcation": _parse_yes_no_unclear(normalize_str(row.get(tri_k, ""))),
                "bifurcation": bif,
                "medina": med,
                "bifurcation_angulation_gt_70": _parse_yes_no_unclear(normalize_str(row.get(ang_k, ""))),
                "aorto_ostial": _parse_yes_no_unclear(normalize_str(row.get(ost_k, ""))),
                "severe_tortuosity": _parse_yes_no_unclear(normalize_str(row.get(tort_k, ""))),
                "thrombus": _parse_yes_no_unclear(normalize_str(row.get(thr_k, ""))),
                "evidence_images": [],
            }
        )
    return {"lesion_attributes": attrs}


def gt_step04_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    s = normalize_str(row.get("SYNTAX Score", "0"))
    try:
        score_int = int(round(float(s)))
    except Exception:
        score_int = 0

    if score_int >= 33:
        cls = "High"
    elif score_int >= 23:
        cls = "Intermediate"
    else:
        cls = "Low"
    return {"syntax_score_est": score_int, "syntax_class": cls}


def build_steps_gt_from_row(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    step01 = gt_step01_from_row(row)
    step02 = gt_step02_from_row(row)
    n = int(step02.get("lesion_count_est", 0))
    return {
        "step00": gt_step00_from_row(row),
        "step01": step01,
        "step02": step02,
        "step025": step02,
        "step026": gt_step026_from_row(row, n),
        "step03": gt_step03_from_row(row, n),
        "step04": gt_step04_from_row(row),
    }


def gt_subset_for_correction(step_name: str, gt_step: Dict[str, Any]) -> Dict[str, Any]:
    allow = {
        "step00": ["per_image_primary_vessel"],
        "step01": ["dominance_label", "pda_source"],
        "step02": ["lesion_count_est", "lesions"],
        "step025": ["lesion_count_est", "lesions"],
        "step026": ["heavy_calcification_by_lesion"],
        "step03": ["lesion_attributes"],
        "step04": ["syntax_score_est", "syntax_class"],
    }[step_name]
    return {k: gt_step.get(k) for k in allow if k in gt_step}


# GT checks
def compare_step00(pred: Dict[str, Any], gt: Dict[str, Any]) -> Tuple[bool, str]:
    g = gt.get("per_image_primary_vessel", [])
    if isinstance(g, list) and len(g) == 0:
        return True, "GT has no step00; skipped."
    p = pred.get("per_image_primary_vessel", [])
    if not isinstance(p, list) or not isinstance(g, list):
        return False, "per_image_primary_vessel missing/invalid."
    return True, "OK"


def compare_step01(pred: Dict[str, Any], gt: Dict[str, Any]) -> Tuple[bool, str]:
    for k in ("dominance_label", "pda_source"):
        if k not in pred:
            return False, f"Missing key: {k}"
        if _lower(pred.get(k)) != _lower(gt.get(k)):
            return False, f"Mismatch {k}: pred={pred.get(k)} gt={gt.get(k)}"
    return True, "OK"


def _canon_segments_only(les: Dict[str, Any]) -> List[str]:
    segs = les.get("segment_numbers", [])
    if not isinstance(segs, list):
        return []
    return sorted([normalize_str(x) for x in segs if normalize_str(x)])


def compare_step02_like(pred: Dict[str, Any], gt: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(pred.get("lesions"), list):
        return False, "Missing/invalid lesions list."
    if pred.get("lesion_count_est") != gt.get("lesion_count_est"):
        return False, f"Mismatch lesion_count_est: pred={pred.get('lesion_count_est')} gt={gt.get('lesion_count_est')}"

    pseg = [_canon_segments_only(x)
            for x in pred["lesions"] if isinstance(x, dict)]
    gseg = [_canon_segments_only(x) for x in gt.get(
        "lesions", []) if isinstance(x, dict)]

    pc = Counter([json.dumps(x, ensure_ascii=False) for x in pseg])
    gc = Counter([json.dumps(x, ensure_ascii=False) for x in gseg])

    if pc != gc:
        only_p = list((pc - gc).elements())
        only_g = list((gc - pc).elements())
        return False, json.dumps(
            {
                "segments_only_in_pred": [json.loads(x) for x in only_p[:10]],
                "segments_only_in_gt": [json.loads(x) for x in only_g[:10]],
            },
            ensure_ascii=False,
        )
    return True, "OK"


def compare_step026(pred: Dict[str, Any], gt: Dict[str, Any]) -> Tuple[bool, str]:
    def build_map(obj: Any) -> Dict[str, str]:
        out = {}
        if isinstance(obj, list):
            for it in obj:
                if isinstance(it, dict) and "lesion_id" in it and "heavy_calcification" in it:
                    out[normalize_str(it["lesion_id"])] = _lower(
                        it["heavy_calcification"])
        return out

    pm = build_map(pred.get("heavy_calcification_by_lesion"))
    gm = build_map(gt.get("heavy_calcification_by_lesion"))
    if pm != gm:
        return False, f"Mismatch heavy_calcification_by_lesion map: pred={pm} gt={gm}"
    return True, "OK"


def compare_step03(pred: Dict[str, Any], gt: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(pred.get("lesion_attributes"), list):
        return False, "Missing/invalid lesion_attributes."
    if not isinstance(gt.get("lesion_attributes"), list):
        return False, "GT missing/invalid lesion_attributes."

    def canon_attr(a: Dict[str, Any]) -> Dict[str, Any]:
        ocf = a.get("occlusion_features", {})
        if not isinstance(ocf, dict):
            ocf = {}
        return {
            "lesion_id": normalize_str(a.get("lesion_id")),
            "total_occlusion": _lower(a.get("total_occlusion")),
            "occlusion_features": {
                "occlusion_age_ge_3_months": _lower(ocf.get("occlusion_age_ge_3_months")),
                "blunt_stump": _lower(ocf.get("blunt_stump")),
                "bridging_collaterals": _lower(ocf.get("bridging_collaterals")),
                "side_branch_at_occlusion": _lower(ocf.get("side_branch_at_occlusion")),
                "first_segment_beyond_occlusion_visualized": _lower(
                    ocf.get("first_segment_beyond_occlusion_visualized")
                ),
            },
            "trifurcation": _lower(a.get("trifurcation")),
            "bifurcation": _lower(a.get("bifurcation")),
            "medina": normalize_str(a.get("medina")),
            "bifurcation_angulation_gt_70": _lower(a.get("bifurcation_angulation_gt_70")),
            "aorto_ostial": _lower(a.get("aorto_ostial")),
            "severe_tortuosity": _lower(a.get("severe_tortuosity")),
            "thrombus": _lower(a.get("thrombus")),
        }

    pred_attrs = (canon_attr(item) for item in pred["lesion_attributes"]
                  if isinstance(item, dict))
    gt_attrs = (canon_attr(item) for item in gt["lesion_attributes"]
                if isinstance(item, dict))
    pmap = {attr["lesion_id"]: attr for attr in pred_attrs}
    gmap = {attr["lesion_id"]: attr for attr in gt_attrs}

    if pmap != gmap:
        diffs = []
        for lid in sorted(set(pmap.keys()) | set(gmap.keys())):
            if pmap.get(lid) != gmap.get(lid):
                diffs.append(
                    {"lesion_id": lid, "pred": pmap.get(lid), "gt": gmap.get(lid)})
            if len(diffs) >= 5:
                break
        return False, json.dumps({"mismatches_first5": diffs}, ensure_ascii=False)

    return True, "OK"


def compare_step04(pred: Dict[str, Any], gt: Dict[str, Any]) -> Tuple[bool, str]:
    if pred.get("syntax_score_est") != gt.get("syntax_score_est"):
        return False, f"Mismatch syntax_score_est: pred={pred.get('syntax_score_est')} gt={gt.get('syntax_score_est')}"
    if normalize_str(pred.get("syntax_class")) != normalize_str(gt.get("syntax_class")):
        return False, f"Mismatch syntax_class: pred={pred.get('syntax_class')} gt={gt.get('syntax_class')}"
    return True, "OK"


def compare_for_step(step_name: str, pred: Dict[str, Any], gt: Dict[str, Any]) -> Tuple[bool, str]:
    if step_name == "step00":
        return compare_step00(pred, gt)
    if step_name == "step01":
        return compare_step01(pred, gt)
    if step_name in ("step02", "step025"):
        return compare_step02_like(pred, gt)
    if step_name == "step026":
        return compare_step026(pred, gt)
    if step_name == "step03":
        return compare_step03(pred, gt)
    if step_name == "step04":
        return compare_step04(pred, gt)
    return False, f"Unknown step: {step_name}"


# GT refinement
SYSTEM_TEXT = (
    "You are a careful assistant. Follow the user's output format rules exactly. "
    "When the user asks for JSON-only, output only a single JSON object and nothing else."
)

CORRECTION_PREFIX = (
    "Your answer for this step is NOT correct.\n"
    "Ground truth for THIS STEP ONLY (current patient) is:\n"
    "{GT_JSON}\n\n"
    "Mismatch summary (what is wrong in your output):\n"
    "{DIFF_SUMMARY}\n\n"
    "Please re-do this step using the same images/context and output EXACTLY ONE JSON object "
    "following the step's OUTPUT FORMAT. Do not add any extra text."
)

# GT prompt near turn limit
FORCED_GT_PREFIX = (
    "WARNING: We are approaching the maximum turn budget for this patient.\n"
    "You MUST follow the ground truth for THIS STEP ONLY exactly.\n"
    "Ground truth is:\n"
    "{GT_JSON}\n\n"
    "You MUST output a JSON object that matches this ground truth exactly for the checked fields.\n"
    "Now re-do this step using the same images/context and output EXACTLY ONE JSON object "
    "following the step's OUTPUT FORMAT. Do not add any extra text."
)


def run_step_with_gt_loop(
    client: OpenAI,
    step_name: str,
    step_template: str,
    send_images: bool,
    images: List[str],
    context: Dict[str, Any],
    gt_step: Dict[str, Any],
    out_dir: str,
    turn_start: int,
    patient_turn_limit: int,
    warn_turn_at: int,
    patient_id_for_logs: str,
) -> Tuple[Dict[str, Any], int]:
    schema = schema_for_step(step_name, len(images))
    max_tokens = MAX_TOKENS_STEP00 if step_name == "step00" else MAX_TOKENS_JSON_TASK

    messages: List[Dict[str, Any]] = [build_system_message(SYSTEM_TEXT)]
    base_prompt = render_prompt(
        step_template, num_images=len(images), context=context)

    attempt = 0
    turn = turn_start
    last_diff = "N/A"
    warned = False

    while True:
        # Turn-limit warning
        if (not warned) and (turn >= warn_turn_at):
            print(
                f"[WARN] {patient_id_for_logs}: turn {turn} >= {warn_turn_at}, limit={patient_turn_limit}")
            warned = True

        if turn > patient_turn_limit:
            raise RuntimeError(
                f"{patient_id_for_logs}: exceeded max turns per patient ({patient_turn_limit}). "
                f"Last step={step_name} last_diff={last_diff}"
            )

        # Forced GT correction
        if turn >= warn_turn_at:
            gt_subset = gt_subset_for_correction(step_name, gt_step)
            user_text = (
                FORCED_GT_PREFIX.format(
                    GT_JSON=json.dumps(
                        gt_subset, ensure_ascii=False, indent=2),
                )
                + "\n\n"
                + base_prompt
            )
        elif attempt == 0:
            user_text = base_prompt
        else:
            gt_subset = gt_subset_for_correction(step_name, gt_step)
            user_text = (
                CORRECTION_PREFIX.format(
                    GT_JSON=json.dumps(
                        gt_subset, ensure_ascii=False, indent=2),
                    DIFF_SUMMARY=last_diff,
                )
                + "\n\n"
                + base_prompt
            )

        if send_images:
            messages.append(build_user_message_with_images(user_text, images))
        else:
            messages.append(build_user_text_message(user_text))

        raw = chat_once(client, messages, max_tokens=max_tokens, schema=schema)
        messages.append(build_assistant_message(raw))

        save_text(os.path.join(out_dir, f"turn_{turn:03d}_raw.txt"), raw)

        pred_obj = parse_json_strict(raw, client, messages, schema)
        save_json(os.path.join(out_dir, f"turn_{turn:03d}.json"), pred_obj)

        ok, diff = compare_for_step(step_name, pred_obj, gt_step)
        if ok:
            save_json(os.path.join(
                out_dir, f"{step_name}_final.json"), pred_obj)
            return pred_obj, turn + 1

        last_diff = diff
        attempt += 1
        turn += 1


# Patients
def list_patient_folders(data_root: str) -> List[str]:
    if not os.path.isdir(data_root):
        raise RuntimeError(f"DATA_ROOT is not a directory: {data_root}")

    names = [name for name in os.listdir(data_root)
             if os.path.isdir(os.path.join(data_root, name))]
    return sorted(names, key=_natural_sort_key)


# CLI
def _parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_folder", default=None)
    return vars(parser.parse_args())


# Main
def main() -> None:
    args = _parse_args()

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)

    step00_path = os.path.join(PROMPT_DIR, PROMPT_FILE_STEP00)
    step00_template = load_text(step00_path)

    gt_path = find_gt_path()
    rows = load_gt_rows(gt_path)

    if args["patient_folder"]:
        patient_folders = [args["patient_folder"]]
    else:
        patient_folders = list_patient_folders(DATA_ROOT)

    steps: List[Tuple[str, str, bool]] = [
        ("step00", step00_template, True),
        ("step01", STEP01_TEMPLATE, True),
        ("step02", STEP02_TEMPLATE, True),
        ("step025", STEP025_TEMPLATE, True),
        ("step026", STEP026_TEMPLATE, True),
        ("step03", STEP03_TEMPLATE, True),
        ("step04", STEP04_TEMPLATE, False),
    ]

    for pf in patient_folders:
        global PATIENT_FOLDER
        PATIENT_FOLDER = pf

        patient_path = os.path.join(DATA_ROOT, PATIENT_FOLDER)
        if not os.path.isdir(patient_path):
            continue

        images = collect_overlay_images(patient_path, MAX_IMAGES)
        if not images:
            print(f"[SKIP] No overlay images: {patient_path}")
            continue

        out_dir = os.path.join(OUTPUT_ROOT, PATIENT_FOLDER, "auto")
        ensure_dir(out_dir)

        try:
            row = get_patient_row(rows, PATIENT_FOLDER)
        except Exception as e:
            print(f"[SKIP] Missing GT for {PATIENT_FOLDER}: {e}")
            continue

        steps_gt = build_steps_gt_from_row(row)

        context: Dict[str, Any] = {
            "patient_folder": PATIENT_FOLDER, "num_images": len(images)}

        turn = 1
        try:
            for step_name, template, send_images in steps:
                validated, turn = run_step_with_gt_loop(
                    client=client,
                    step_name=step_name,
                    step_template=template,
                    send_images=send_images,
                    images=images,
                    context=context,
                    gt_step=steps_gt[step_name],
                    out_dir=out_dir,
                    turn_start=turn,
                    patient_turn_limit=MAX_TURNS_PER_PATIENT,
                    warn_turn_at=WARN_TURN_AT,
                    patient_id_for_logs=PATIENT_FOLDER,
                )
                context[step_name] = validated
        except Exception as e:
            # Record failed patient
            save_text(os.path.join(out_dir, "FAILED.txt"), str(e) + "\n")
            print(f"[FAIL] {PATIENT_FOLDER}: {e}")
            continue

        save_text(
            os.path.join(out_dir, "session_meta.txt"),
            f"turns={turn-1}\npatient_folder={PATIENT_FOLDER}\nnum_images={len(images)}\ngt_path={gt_path}\n",
        )
        print(f"[DONE] {PATIENT_FOLDER} -> {out_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
