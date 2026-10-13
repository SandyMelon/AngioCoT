"""Generate rationales from final states."""

import os
import re
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# Config

# Model server.
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
BASE_URL = DEFAULT_BASE_URL

MODEL_PATH = "your-served-model-name"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-used")

# Generation settings.
CHAT_KWARGS = {
    "max_tokens": int(os.getenv("COT_MAX_TOKENS", "2000")),
    "temperature": float(os.getenv("COT_TEMPERATURE", "0.0")),
    "frequency_penalty": 0.0,
}

# Output filenames
OUT_RAW = "cot_raw.txt"
OUT_RAW_FIX = "cot_raw_fix.txt"
OUT_JSON = "cot.json"

# Excluded filenames.
EXCLUDE_NAME_CONTAINS = ["cot_raw", "cot.json", "final_answer", "cot_markdown"]

# Final states.
FINAL_FILE_REGEX = re.compile(r"^step\d+_final\.json$", re.IGNORECASE)

# File helpers


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=True, indent=2))


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
                    return s[start: i + 1]
    return ""


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()

    # Direct parse
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
        return {"_value": obj}
    except Exception:
        pass

    # Extract JSON from mixed output.
    extracted = extract_first_json_object(t)
    if not extracted:
        return None
    try:
        obj = json.loads(extracted)
        if isinstance(obj, dict):
            return obj
        return {"_value": obj}
    except Exception:
        return None


def is_valid_cot_obj(obj: Optional[Dict[str, Any]]) -> bool:
    return (isinstance(obj, dict)
            and list(obj) == ["cot"]
            and isinstance(obj["cot"], str))


# Model client

SYSTEM_PROMPT = (
    "You are an assistant that MUST follow output constraints exactly.\n"
    "Never reveal internal deliberation. Provide only concise, user-facing rationale.\n"
    "If the user asks for JSON only, output JSON only.\n"
)

# Excluded phrases.
BANNED_PHRASES = [
    "let's think",
    "let me think",
    "we are given",
    "step by step",
    "chain of thought",
    "thought process",
    "i will",
    "we must",
]


def _safe_json_mode_kwargs() -> Dict[str, Any]:
    # JSON output mode.
    return {"response_format": {"type": "json_object"}}


def chat_once(client: OpenAI, messages: List[Dict[str, Any]], use_json_mode: bool = True) -> str:
    kwargs: Dict[str, Any] = {
        "model": MODEL_PATH,
        "messages": messages,
        **CHAT_KWARGS,
    }

    if use_json_mode:
        kwargs.update(_safe_json_mode_kwargs())

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception:
        # Retry without JSON mode.
        if use_json_mode:
            kwargs.pop("response_format", None)
            resp = client.chat.completions.create(**kwargs)
        else:
            raise

    msg = resp.choices[0].message
    content = msg.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content)


# Prompt helpers


def compact_json(obj: Any) -> str:
    # Compact to save tokens
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"))


def merge_context_items(context_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge state data; later keys take precedence."""
    merged: Dict[str, Any] = {}
    for item in context_items:
        data = item.get("data")
        if isinstance(data, dict):
            merged.update(data)
    return merged


def build_cot_prompt(context_items: List[Dict[str, Any]], *, merge: bool = False) -> str:
    """Build the rationale prompt, optionally merging state data."""
    if merge:
        context_json = compact_json([{"data": merge_context_items(context_items)}])
    else:
        context_json = compact_json(context_items)

    prompt = (
        "TASK\n"
        "Using ONLY the JSON objects in CONTEXT, write a concise clinician-style rationale.\n"
        "Do NOT add new lesions, segments, devices, measurements, history, or facts not present in CONTEXT.\n"
        "If a detail is missing, say it is not determinable from CONTEXT.\n"
        "If multiple items conflict, prefer the later item in CONTEXT.\n\n"
        "STYLE\n"
        "- Produce exactly 10 numbered steps, from 1) to 10), each no more than 25 words.\n"
        "- Use this fixed step order:\n"
        "  1) Image quality and adequacy for assessment.\n"
        "  2) Coronary anatomy and vessel visualization.\n"
        "  3) Coronary dominance.\n"
        "  4) Lesion inventory and total lesion count.\n"
        "  5) Lesion locations and stenosis significance.\n"
        "  6) Lesion length and heavy calcification.\n"
        "  7) Complex lesion features: occlusion, thrombus, bifurcation, "
        "trifurcation, ostial involvement, angulation, and tortuosity.\n"
        "  8) Diffuse disease and small vessel involvement.\n"
        "  9) SYNTAX score and complexity class.\n"
        "  10) Final summary and justification.\n"
        "- Adopt a clinician’s tone: state the evidence considered and the intermediate conclusion drawn.\n"
        "- Explicitly rule in or rule out hypotheses by referencing both positive and negative findings.\n"
        "- Conclude with a step summarising the final answer and its justification.\n"
        "- No meta talk. Do NOT mention files, turns, prompts, instructions, or 'context'.\n"
        "- Do NOT use any of these phrases (case-insensitive): "
        + ", ".join([f'\"{p}\"' for p in BANNED_PHRASES])
        + ".\n"
        "- No preambles, no disclaimers.\n\n"
        "CONTEXT\n"
        f"{context_json}\n\n"
        "OUTPUT FORMAT\n"
        "Return ONE JSON object with EXACTLY this key:\n"
        "- cot: a single string containing the numbered steps.\n\n"
        "STRICT OUTPUT RULES\n"
        "- Output ONLY the JSON object.\n"
        "- The first non-whitespace character MUST be '{'.\n"
        "- Do not add any other top-level keys.\n"
    )
    return prompt


def build_messages(prompt_text: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]


def repair_prompt(reason: str) -> str:
    return (
        "Rewrite as JSON ONLY.\n"
        f"Problem: {reason}\n"
        "The cot string must contain exactly 10 numbered steps, from 1) to 10), "
        "in the originally requested order.\n"
        "Output exactly: {\"cot\":\"1) ...\\n2) ...\"}\n"
        "No other keys. No extra text.\n"
    )


def contains_banned_phrase(text: str) -> Optional[str]:
    low = (text or "").lower()
    for p in BANNED_PHRASES:
        if p in low:
            return p
    return None


# State loading


def should_exclude_name(name: str) -> bool:
    low = name.lower()
    if "raw" in low:
        return True
    for s in EXCLUDE_NAME_CONTAINS:
        if s in low:
            return True
    return False


def _step_index_from_final_name(name: str) -> Tuple[int, str]:
    """Extract the numeric step index."""
    m = re.match(r"^step(\d+)_final\.json$", name.lower())
    if not m:
        return (10**9, name)
    return (int(m.group(1)), name)


def load_context_from_dir(folder: str) -> List[Dict[str, Any]]:
    """Load final state files."""
    if not os.path.isdir(folder):
        raise RuntimeError(f"Not a directory: {folder}")

    names = [n for n in os.listdir(folder) if FINAL_FILE_REGEX.match(n)]
    names.sort(key=_step_index_from_final_name)

    context_items: List[Dict[str, Any]] = []
    for n in names:
        path = os.path.join(folder, n)
        try:
            text = read_text(path)
        except Exception:
            continue

        obj = try_parse_json(text)
        if obj is None:
            continue

        # Skip rationale files.
        if is_valid_cot_obj(obj):
            continue

        step_num, _ = _step_index_from_final_name(n)
        context_items.append(
            {
                "filename": n,
                "turn": None,          # Final files have no turn.
                "step": None if step_num >= 10**9 else step_num,
                "data": obj,
            }
        )

    if not context_items:
        raise RuntimeError(
            "No readable step*_final.json files found in the folder.")
    return context_items


# Main

# Structured-state output root.
DATA_ROOT_DIR = os.path.abspath("/path/to/your/structured_cot_directory")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a clinician-style CoT rationale from step*_final.json outputs.")
    ap.add_argument(
        "--merge-context",
        action="store_true",
        help="Merge all context items into one dictionary before prompting the model. Later keys override earlier ones.",
    )
    args = ap.parse_args()

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL, timeout=600.0)

    failed_cases: List[str] = []

    # Patient folders.
    try:
        patient_names = os.listdir(DATA_ROOT_DIR)
    except Exception:
        patient_names = []

    patient_names.sort()

    for patient_name in patient_names:
        patient_dir = os.path.join(DATA_ROOT_DIR, patient_name)
        if not os.path.isdir(patient_dir):
            continue

        auto_dir = os.path.join(patient_dir, "auto")
        if not os.path.isdir(auto_dir):
            failed_cases.append(patient_name)
            continue

        try:
            folder = os.path.abspath(auto_dir)
            context_items = load_context_from_dir(folder)

            prompt = build_cot_prompt(context_items, merge=args.merge_context)
            messages = build_messages(prompt)

            raw = chat_once(client, messages, use_json_mode=True)
            write_text(os.path.join(folder, OUT_RAW), raw)

            parsed = try_parse_json(raw)

            # Check format and phrases.
            if is_valid_cot_obj(parsed):
                bad = contains_banned_phrase(parsed["cot"])
                if bad is None:
                    write_json(os.path.join(folder, OUT_JSON), parsed)
                    continue
                parsed = None  # force repair

            # Format repair.
            last_raw = raw
            for _ in range(3):
                reason = "Not a single JSON object with only key 'cot'."
                if parsed is not None and isinstance(parsed, dict) and "cot" in parsed:
                    reason = "Contains banned phrase or invalid structure."

                fix_user = repair_prompt(reason)
                fix_messages = messages + [
                    {"role": "assistant", "content": last_raw},
                    {"role": "user", "content": fix_user},
                ]
                fixed_raw = chat_once(client, fix_messages, use_json_mode=True)
                write_text(os.path.join(folder, OUT_RAW_FIX), fixed_raw)

                parsed = try_parse_json(fixed_raw)
                if (is_valid_cot_obj(parsed)
                        and contains_banned_phrase(parsed["cot"]) is None):
                    write_json(os.path.join(folder, OUT_JSON), parsed)
                    break

                last_raw = fixed_raw
            else:
                # Save fallback output.
                fallback_text = (extract_first_json_object(
                    last_raw) or last_raw or "").strip()
                fallback_text = re.sub(r"\s+", " ", fallback_text).strip()
                write_json(os.path.join(folder, OUT_JSON), {"cot": fallback_text})
                failed_cases.append(patient_name)

        except Exception:
            failed_cases.append(patient_name)

    print(failed_cases)


if __name__ == "__main__":
    main()
