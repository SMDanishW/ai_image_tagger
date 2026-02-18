"""
Shirt Design Tagger — OpenAI + Gemini
======================================
Reads images from ./artwork_images/ and a master table (CSV or XLSX),
tags each image with both OpenAI and Gemini, and outputs a merged CSV.

Outputs:
  - openai_tagged_master_table_jns.csv   (intermediary — OpenAI results)
  - gemini_tagged_master_table_jns.csv   (intermediary — Gemini results)
  - tagged_master_table_jns.csv          (final merged output)

Usage:
  1. Place artwork images in ./artwork_images/
  2. Place master_table.csv or master_table.xlsx in the working directory
  3. Create a .env file with OPENAI_API_KEY and GEMINI_API_KEY
  4. Run:  python shirt_design_tagger.py
"""

import base64
import json
import os
import sys
import time
import pandas as pd
from pathlib import Path
from PIL import Image

# ── Optional: load .env file ──────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env not required if env vars are already set

# ============================================================
# ⚙️  CONFIGURATION
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

IMAGE_FOLDER = "./artwork_images"
OPENAI_MODEL = "gpt-5-mini-2025-08-07"              # Change as needed (gpt-4o, gpt-4.1, etc.)
GEMINI_MODEL = "gemini-2.5-flash"

OUTPUT_FINAL = "tagged_master_table_jns.csv"
OUTPUT_OPENAI = "openai_tagged_master_table_jns.csv"
OUTPUT_GEMINI = "gemini_tagged_master_table_jns.csv"

# Delay between API calls (seconds) to respect rate limits
API_DELAY = 1.0

# Schema file — client-editable CSV or XLSX with columns: Theme, Motifs, Style
SCHEMA_FILE_CSV = "tagging_schema.csv"
SCHEMA_FILE_XLSX = "tagging_schema.xlsx"

# ============================================================
# 📋  TAGGING SCHEMA — loaded dynamically from file
# ============================================================

def load_tagging_schema() -> dict:
    """
    Load the tagging schema from tagging_schema.csv or tagging_schema.xlsx.
    Expected format: 3 columns (Theme, Motifs, Style) with allowed values
    listed as rows. Columns may have unequal lengths (empty cells are ignored).
    """
    csv_path = Path(SCHEMA_FILE_CSV)
    xlsx_path = Path(SCHEMA_FILE_XLSX)

    if csv_path.exists():
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        print(f"📋 Loaded schema from {csv_path}")
    elif xlsx_path.exists():
        df = pd.read_excel(xlsx_path)
        print(f"📋 Loaded schema from {xlsx_path}")
    else:
        print(f"❌ ERROR: Neither {SCHEMA_FILE_CSV} nor {SCHEMA_FILE_XLSX} found!")
        sys.exit(1)

    # Normalize column names
    df.columns = df.columns.str.strip()

    schema = {}
    for col in df.columns:
        values = df[col].dropna().astype(str).str.strip().tolist()
        values = [v for v in values if v]  # remove empty strings
        schema[col] = values

    # Validate expected columns exist
    for expected in ["Theme", "Motifs", "Style"]:
        if expected not in schema:
            # Try case-insensitive match
            matched = [k for k in schema if k.lower() == expected.lower()]
            if matched:
                schema[expected] = schema.pop(matched[0])
            else:
                print(f"⚠️  Column '{expected}' not found in schema file. Using empty list.")
                schema[expected] = []

    for key, values in schema.items():
        print(f"   {key}: {len(values)} options")

    return schema


TAGGING_SCHEMA = load_tagging_schema()

# ============================================================
# 🎯  FEW-SHOT EXAMPLES
# ============================================================
FEW_SHOT_EXAMPLES = [
    {"image_path": "./jns_assets/artwork_870.png", "print_id": "Artwork 870",
     "theme": "Ski", "motifs": "skiing, flowers, boot, Aspen", "style": "retro, vintage"},
    {"image_path": "./jns_assets/artwork_871.png", "print_id": "Artwork 871",
     "theme": "Western", "motifs": "cowboy, boot, flowers, Winter Park", "style": "folk, vintage"},
    {"image_path": "./jns_assets/artwork_872.png", "print_id": "Artwork 872",
     "theme": "Outdoors", "motifs": "hiking, camping, flowers, boot", "style": "retro, vintage"},
]

# ============================================================
# 📝  SYSTEM PROMPT BUILDER
# ============================================================
TAGGING_EXAMPLES_TEXT = [
    {"print_id": "Artwork 870", "theme": "Ski", "motifs": "skiing, flowers, boot, Aspen", "style": "retro, vintage"},
    {"print_id": "Artwork 871", "theme": "Western", "motifs": "cowboy, boot, flowers, Winter Park", "style": "folk, vintage"},
    {"print_id": "Artwork 872", "theme": "Outdoors", "motifs": "hiking, camping, flowers, boot", "style": "retro, vintage"},
    {"print_id": "Artwork 873", "theme": "Western", "motifs": "buffalo, Wyoming, cabin, mountains", "style": "line art, folk"},
    {"print_id": "Artwork 874", "theme": "Outdoors", "motifs": "mountain, bird, hawk, eagle, hiking, camping, Glacier National Park", "style": "line art, folk"},
    {"print_id": "Artwork 875", "theme": "Tropics", "motifs": "beach, ocean, surfing, water, Palm Coast", "style": "line art, retro, Mid Century"},
    {"print_id": "Artwork 876", "theme": "Outdoors", "motifs": "flowers, Alaska, frontier", "style": "line art, folk"},
    {"print_id": "Artwork 877", "theme": "Mountains", "motifs": "moose, mountain, Montana", "style": "line art, folk"},
    {"print_id": "Artwork 878", "theme": "Mountains", "motifs": "moose, mountain, Montana", "style": "line art, folk"},
    {"print_id": "Artwork 879", "theme": "Mountains", "motifs": "mountain, pine trees, Whitefish Montana, sun, moon", "style": "line art, folk"},
    {"print_id": "Artwork 880", "theme": "Mountains", "motifs": "pine tree, mountains, Yellowstone National Park", "style": "retro, vintage"},
    {"print_id": "Artwork 881", "theme": "Tropics", "motifs": "palm tree, beach, San Diego", "style": "retro, Mid Century"},
    {"print_id": "Artwork 882", "theme": "Ski", "motifs": "skiing, skiier, mountain, Stowe Vermont", "style": "retro, vintage"},
    {"print_id": "Artwork 883", "theme": "Mountains", "motifs": "bike, moose, mountain, pine tree, Whitefish Montana", "style": "line art, folk, whimsical"},
    {"print_id": "Artwork 884", "theme": "Outdoors", "motifs": "pine tree, big foot, sasquatch, cryptid", "style": "line art, whimsical"},
    {"print_id": "Artwork 885", "theme": "Outdoors", "motifs": "big foot, sasquatch, cryptid", "style": "folk, minimalist"},
    {"print_id": "Artwork 886", "theme": "Lakes & Rivers", "motifs": "fish, fishhook, fishing, trout", "style": "line art"},
    {"print_id": "Artwork 887", "theme": "Nautical", "motifs": "ocean, sun, waves, water, Bermuda", "style": "Ukiyo-e, retro"},
    {"print_id": "Artwork 888", "theme": "Tropics", "motifs": "palm tree, sun, ocean, beach", "style": "retro, minimalist"},
]


def build_system_prompt():
    themes_list = ", ".join(TAGGING_SCHEMA.get('Theme', []))
    motifs_list = ", ".join(TAGGING_SCHEMA.get('Motifs', []))
    styles_list = ", ".join(TAGGING_SCHEMA.get('Style', []))

    few_shot_text = ""
    for ex in TAGGING_EXAMPLES_TEXT:
        few_shot_text += f"""
            Print ID: {ex['print_id']}
            Theme: {ex['theme']}
            Motifs: {ex['motifs']}
            Style: {ex['style']}"""

    #System prompt by Gemini
    return f"""Role: Expert Visual Tagger for Jack & Sage (Premium Outdoor Apparel).
        Task: Image analysis and metadata generation for Adobe Creative Cloud / Print-on-Demand.

        [STRICT CONSTRAINTS]
        1. OUTPUT: Return ONLY valid JSON. No markdown code blocks (no ```json). No conversational filler.
        2. THEME: Select exactly ONE primary environment or activity from the provided schema.
        3. MOTIFS & STYLE LIMIT: The total combined character count for the "motifs" and "style" strings must not exceed 130 characters. 
        4. MOTIFS CONTENT: Prioritize specific nouns (e.g., "Grizzly Bear" vs "Animal"). Transcribe all visible text (Parks, States, Brands).
        5. STYLE CONTENT: Describe visual execution only (e.g., "Vintage," "Etching," "Badge Logo").

        [BRAND IDENTITY]
        Jack & Sage is a premium, nostalgic, adventure-driven outdoor brand. Focus on National Parks, Alpine, Coastal, and Western aesthetics.

        [SCHEMA]
        Themes: {themes_list}
        Motifs: {motifs_list}
        Styles: {styles_list}

        [FEW-SHOT EXAMPLES]
        {few_shot_text}

        [JSON STRUCTURE]
        {{
        "print_id": "string",
        "theme": "string",
        "motifs": "string",
        "style": "string"}}
        """

    # return f"""You are an expert visual tagger for a print-on-demand shirt company called Jack & Sage.
    # Your job: look at a shirt design image and classify it into 3 categories.
    # ## TAGGING SCHEMA (preferred values, but you may add freeform descriptors when needed)
    # **Theme** (pick ONE primary theme):
    # {', '.join(TAGGING_SCHEMA['Theme'])}

    # **Motifs** (list ALL visible motifs: include location names if text is visible):
    # {', '.join(TAGGING_SCHEMA['Motifs'])}

    # **Style** (pick 1-3 that apply):
    # {', '.join(TAGGING_SCHEMA['Style'])}
    # ## TAGGING RULES
    # 1. Theme: Choose exactly ONE primary theme from the schema. Use 'Other' only as last resort.
    # 2. Motifs: List everything visible: animals, objects, text, place names, symbols. Use schema terms where applicable but ADD freeform terms for anything not listed.
    # 3. Style: Pick from schema terms. You may add freeform descriptors like 'folk', 'Mid Century', 'Ukiyo-e' if they fit.
    # 4. If text/lettering is visible, include location or brand names as motifs.
    # 5. THe number of words in motifs and style should be 130 characters collected since Adobe has this limit for tags
    # ## HUMAN-TAGGED EXAMPLES (learn from these):
    # {few_shot_text}
    # ## OUTPUT FORMAT
    # Respond ONLY with valid JSON - no markdown, no explanation:
    # {{"print_id": "<filename>", "theme": "<single theme>", "motifs": "<comma-separated>", "style": "<comma-separated>"}}
    # """

SYSTEM_PROMPT = build_system_prompt()

# ============================================================
# 🛠️  HELPER FUNCTIONS
# ============================================================

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_media_type(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    return {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp"}.get(ext, "image/png")


def get_image_files(folder: str) -> list:
    valid_ext = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(
        [f for f in Path(folder).iterdir() if f.suffix.lower() in valid_ext],
        key=lambda x: x.name
    )


def normalize_print_id(raw_id: str) -> str:
    """Normalize print IDs for matching: 'Artwork_870' -> 'artwork 870'"""
    return str(raw_id).strip().lower().replace("_", " ")


def find_image_for_print_id(print_id: str, image_files: list) -> str | None:
    """Find the image file that matches a given Print ID."""
    norm_id = normalize_print_id(print_id)
    for img in image_files:
        if normalize_print_id(img.stem) == norm_id:
            return str(img)
    # Fallback: check if the numeric part matches
    import re
    id_nums = re.findall(r'\d+', str(print_id))
    if id_nums:
        for img in image_files:
            stem_nums = re.findall(r'\d+', img.stem)
            if stem_nums and id_nums[-1] == stem_nums[-1]:
                return str(img)
    return None


# ============================================================
# 🔑  CUSTOM EXCEPTIONS FOR CREDIT EXHAUSTION
# ============================================================

class APICreditsExhausted(Exception):
    """Raised when an API returns a billing / quota / rate-limit error."""
    pass


def is_credit_error(exc: Exception) -> bool:
    """Heuristic check: does the exception look like a billing/quota error?"""
    msg = str(exc).lower()
    indicators = [
        "insufficient_quota", "rate_limit", "billing", "quota",
        "exceeded", "429", "resource_exhausted", "payment",
        "budget", "credit", "limit", "overloaded",
    ]
    return any(ind in msg for ind in indicators)


# ============================================================
# 🤖  OPENAI TAGGER
# ============================================================

def tag_single_image_openai(client, image_path: str, fewshot_examples: list,
                            verbose: bool = True) -> dict:
    """Tag one image using OpenAI vision model."""
    filename = Path(image_path).stem

    if verbose:
        print(f"  🔍 [OpenAI] Tagging: {Path(image_path).name}...", end=" ")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add few-shot examples
    for ex in fewshot_examples:
        ex_path = ex["image_path"]
        if not Path(ex_path).exists():
            continue
        ex_b64 = encode_image_to_base64(ex_path)
        ex_mt = get_media_type(ex_path)
        expected = {
            "print_id": ex["print_id"], "theme": ex["theme"],
            "motifs": ex["motifs"], "style": ex["style"]
        }
        messages.append({
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": f"data:{ex_mt};base64,{ex_b64}"},
                {"type": "input_text", "text": "Few-shot example. Tag this shirt design and return JSON with keys: print_id, theme, motifs, style."},
            ],
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "output_text", "text": json.dumps(expected)}],
        })

    # Target image
    image_data = encode_image_to_base64(image_path)
    media_type = get_media_type(image_path)
    messages.append({
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": f"data:{media_type};base64,{image_data}"},
            {"type": "input_text", "text": f"Now tag this shirt design. Filename: {filename}. Return JSON only."},
        ],
    })

    try:
        response = client.responses.create(model=OPENAI_MODEL, input=messages)
        # Find the actual message output (skip ReasoningItem blocks)
        raw = None
        for item in response.output:
            if item.type == "message" and item.content:
                raw = item.content[0].text
                break
        if raw is None:
            raise Exception(f"No message content in OpenAI response. Output types: {[item.type for item in response.output]}")
    except Exception as e:
        if is_credit_error(e):
            raise APICreditsExhausted(f"OpenAI credits exhausted: {e}")
        raise

    # Parse JSON
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]

    try:
        result = json.loads(raw.strip())
    except json.JSONDecodeError:
        if verbose:
            print(f"⚠️ JSON parse error")
        result = {"print_id": filename, "theme": "PARSE_ERROR",
                  "motifs": raw[:100], "style": ""}

    if verbose:
        print(f"✅ {result.get('theme')} | {result.get('motifs', '')[:50]}...")
    return result


# ============================================================
# 🟢  GEMINI TAGGER
# ============================================================

def tag_single_image_gemini(gemini_model, image_path: str, fewshot_examples: list,
                            verbose: bool = True) -> dict:
    """Tag one image using Gemini vision model."""
    target_img = Image.open(image_path)
    filename = Path(image_path).stem

    if verbose:
        print(f"  🔍 [Gemini] Tagging: {Path(image_path).name}...", end=" ")

    prompt_parts = [SYSTEM_PROMPT]

    # Few-shot examples
    for ex in fewshot_examples:
        ex_path = ex["image_path"]
        if not Path(ex_path).exists():
            continue
        prompt_parts.append(Image.open(ex_path))
        prompt_parts.append(f"Example Tagging for {ex['print_id']}: {json.dumps(ex)}")

    # Target image
    prompt_parts.append(target_img)
    prompt_parts.append(f"Now tag this shirt design. Filename: {filename}. Return JSON only.")

    try:
        response = gemini_model.generate_content(
            prompt_parts,
            generation_config={"response_mime_type": "application/json"}
        )
        raw = response.text
        result = json.loads(raw.strip())
    except Exception as e:
        if is_credit_error(e):
            raise APICreditsExhausted(f"Gemini credits exhausted: {e}")
        if verbose:
            print(f"⚠️ Error: {e}")
        result = {"print_id": filename, "theme": "ERROR",
                  "motifs": str(e)[:100], "style": ""}

    if verbose:
        print(f"✅ {result.get('theme')} | {result.get('motifs', '')[:50]}...")
    return result


# ============================================================
# 📊  LOAD MASTER TABLE
# ============================================================

def load_master_table() -> pd.DataFrame:
    """Load master_table.csv or master_table.xlsx, whichever exists."""
    csv_path = Path("master_table.csv")
    xlsx_path = Path("master_table.xlsx")

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"📄 Loaded {csv_path} ({len(df)} rows)")
    elif xlsx_path.exists():
        df = pd.read_excel(xlsx_path)
        print(f"📄 Loaded {xlsx_path} ({len(df)} rows)")
    else:
        print("⚠️  Neither master_table.csv nor master_table.xlsx found.")
        print("   Generating master_table.csv from images in artwork_images/...")

        image_files = get_image_files(IMAGE_FOLDER)
        if not image_files:
            print("❌ ERROR: No images found in artwork_images/ either. Nothing to process.")
            sys.exit(1)

        # Build Print IDs from filenames: "artwork_870.png" -> "Artwork 870"
        import re
        rows = []
        for img in image_files:
            stem = img.stem  # e.g. "artwork_870" or "Artwork_870"
            # Convert underscores to spaces and title-case for a clean Print ID
            clean_id = stem.replace("_", " ").strip()
            # Title-case only the first word if it looks like "artwork 870"
            parts = clean_id.split(" ", 1)
            clean_id = parts[0].capitalize() + (" " + parts[1] if len(parts) > 1 else "")
            rows.append({
                "Print ID": clean_id,
                "human_Theme": "",
                "human_Motifs": "",
                "human_Style": "",
            })

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"📄 Created {csv_path} ({len(df)} rows from image filenames)")
        print("   Human tag columns are empty — fill them in later if needed.")

    # Normalize column names — find Print ID, Theme, Motifs, Style
    df.columns = df.columns.str.strip()

    # Try to identify the key columns (case-insensitive)
    col_map = {}
    for col in df.columns:
        cl = col.lower().replace("_", " ").strip()
        if "print" in cl and "id" in cl:
            col_map["Print ID"] = col
        elif cl in ("theme", "themes"):
            col_map["human_Theme"] = col
        elif cl in ("motif", "motifs"):
            col_map["human_Motifs"] = col
        elif cl in ("style", "styles"):
            col_map["human_Style"] = col

    if "Print ID" not in col_map:
        print(f"⚠️  Could not find a 'Print ID' column. Available columns: {list(df.columns)}")
        print("    Will use the first column as Print ID.")
        col_map["Print ID"] = df.columns[0]

    # Rename to standard names
    rename = {}
    for standard_name, original_col in col_map.items():
        rename[original_col] = standard_name
    df = df.rename(columns=rename)

    # Ensure human columns exist (fill with empty if not found)
    for c in ["human_Theme", "human_Motifs", "human_Style"]:
        if c not in df.columns:
            df[c] = ""

    print(f"   Columns mapped: {col_map}")
    return df


# ============================================================
# 🔄  LOAD EXISTING INTERMEDIARY RESULTS (for resume)
# ============================================================

def load_existing_results(filepath: str) -> dict:
    """Load already-tagged results from intermediary CSV, keyed by normalized print_id."""
    p = Path(filepath)
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
        results = {}
        for _, row in df.iterrows():
            pid = normalize_print_id(str(row.get("Print ID", row.get("print_id", ""))))
            results[pid] = row.to_dict()
        print(f"   ♻️  Loaded {len(results)} existing results from {filepath}")
        return results
    except Exception as e:
        print(f"   ⚠️  Could not load {filepath}: {e}")
        return {}


# ============================================================
# 🚀  MAIN BATCH PROCESSING
# ============================================================

def run_batch():
    print("=" * 60)
    print("  Shirt Design Tagger — OpenAI + Gemini")
    print("=" * 60)
    print()

    # ── Load master table ──
    master_df = load_master_table()
    image_files = get_image_files(IMAGE_FOLDER)
    print(f"📁 Found {len(image_files)} images in {IMAGE_FOLDER}\n")

    # ── Initialize API clients ──
    openai_client = None
    gemini_model = None

    if OPENAI_API_KEY:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI client initialized")
    else:
        print("⚠️  OPENAI_API_KEY not set — skipping OpenAI tagging")

    if GEMINI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(model_name=GEMINI_MODEL)
        print("✅ Gemini model initialized")
    else:
        print("⚠️  GEMINI_API_KEY not set — skipping Gemini tagging")

    if not openai_client and not gemini_model:
        print("\n❌ No API keys configured. Set OPENAI_API_KEY and/or GEMINI_API_KEY.")
        sys.exit(1)

    # ── Filter few-shot examples to only those with existing images ──
    valid_fewshot = [ex for ex in FEW_SHOT_EXAMPLES if Path(ex["image_path"]).exists()]
    print(f"🎯 {len(valid_fewshot)} few-shot examples with images available\n")

    # ── Load existing intermediary results (for resume after crash) ──
    existing_openai = load_existing_results(OUTPUT_OPENAI)
    existing_gemini = load_existing_results(OUTPUT_GEMINI)

    # ── Process each row in master table ──
    # Filter to only rows that have a matching image in the folder
    rows_with_images = []

    for idx, row in master_df.iterrows():
        print_id = str(row["Print ID"])
        image_path = find_image_for_print_id(print_id, image_files)
        if image_path:
            rows_with_images.append((idx, row, image_path))
        else:
            if verbose := True:
                print(f"  ℹ️  Print ID '{print_id}' has no image in folder — excluded")

    # ─Also include images NOT in master table
    matched_norms = {normalize_print_id(Path(img_path).stem) for _, _, img_path in rows_with_images}

    for img_path in image_files:
        pid_from_file = Path(img_path).stem.replace("_", " ")
        norm_pid = normalize_print_id(pid_from_file)

        if norm_pid in matched_norms:
            continue  # Already matched via master table

        # Check if it belongs to a master row that failed path matching
        master_match = master_df[master_df["Print ID"].apply(normalize_print_id) == norm_pid]
        if not master_match.empty:
            idx = master_match.index[0]
            rows_with_images.append((idx, master_match.iloc[0], str(img_path)))
        else:
            print(f" ➕ Image '{Path(img_path).name}' not in master table — adding with empty human tags")
            fake_row = pd.Series({"Print ID": pid_from_file})
            rows_with_images.append((None, fake_row, str(img_path)))

    matched_norms = None  # free memory

    openai_results = []
    gemini_results = []
    openai_stopped = not bool(openai_client)
    gemini_stopped = not bool(gemini_model)

    total = len(rows_with_images)
    print(f"\n🖼️  {total} images matched from master table (out of {len(master_df)} rows)")

    for seq_idx, (idx, row, image_path) in enumerate(rows_with_images):
        print_id = str(row["Print ID"])
        norm_pid = normalize_print_id(print_id)
        seq = seq_idx + 1

        print(f"\n[{seq}/{total}] Print ID: {print_id}")

        # ── OpenAI ──
        if not openai_stopped:
            if norm_pid in existing_openai:
                print(f"  ♻️  [OpenAI] Using cached result")
                cached = existing_openai[norm_pid]
                openai_results.append({
                    "Print ID": print_id,
                    "openai_Theme": cached.get("openai_Theme", cached.get("theme", "")),
                    "openai_Motifs": cached.get("openai_Motifs", cached.get("motifs", "")),
                    "openai_Style": cached.get("openai_Style", cached.get("style", "")),
                })
            else:
                try:
                    res = tag_single_image_openai(openai_client, image_path,
                                                  valid_fewshot, verbose=True)
                    openai_results.append({
                        "Print ID": print_id,
                        "openai_Theme": res.get("theme", ""),
                        "openai_Motifs": res.get("motifs", ""),
                        "openai_Style": res.get("style", ""),
                    })
                    time.sleep(API_DELAY)
                except APICreditsExhausted as e:
                    print(f"\n🛑 OPENAI CREDITS EXHAUSTED: {e}")
                    print("   Saving intermediary OpenAI results and stopping OpenAI calls...")
                    openai_results.append({
                        "Print ID": print_id, "openai_Theme": "CREDIT_ERROR",
                        "openai_Motifs": str(e)[:200], "openai_Style": ""
                    })
                    openai_stopped = True
                    # Save intermediary immediately
                    _save_intermediary(master_df, openai_results, "openai", OUTPUT_OPENAI)
                except Exception as e:
                    print(f"  ❌ [OpenAI] Unexpected error: {e}")
                    openai_results.append({
                        "Print ID": print_id, "openai_Theme": "ERROR",
                        "openai_Motifs": str(e)[:200], "openai_Style": ""
                    })
        else:
            openai_results.append({
                "Print ID": print_id, "openai_Theme": "SKIPPED",
                "openai_Motifs": "", "openai_Style": ""
            })

        # ── Gemini ──
        if not gemini_stopped:
            if norm_pid in existing_gemini:
                print(f"  ♻️  [Gemini] Using cached result")
                cached = existing_gemini[norm_pid]
                gemini_results.append({
                    "Print ID": print_id,
                    "gemini_Theme": cached.get("gemini_Theme", cached.get("theme", "")),
                    "gemini_Motifs": cached.get("gemini_Motifs", cached.get("motifs", "")),
                    "gemini_Style": cached.get("gemini_Style", cached.get("style", "")),
                })
            else:
                try:
                    res = tag_single_image_gemini(gemini_model, image_path,
                                                  valid_fewshot, verbose=True)
                    gemini_results.append({
                        "Print ID": print_id,
                        "gemini_Theme": res.get("theme", ""),
                        "gemini_Motifs": res.get("motifs", ""),
                        "gemini_Style": res.get("style", ""),
                    })
                    time.sleep(API_DELAY)
                except APICreditsExhausted as e:
                    print(f"\n🛑 GEMINI CREDITS EXHAUSTED: {e}")
                    print("   Saving intermediary Gemini results and stopping Gemini calls...")
                    gemini_results.append({
                        "Print ID": print_id, "gemini_Theme": "CREDIT_ERROR",
                        "gemini_Motifs": str(e)[:200], "gemini_Style": ""
                    })
                    gemini_stopped = True
                    _save_intermediary(master_df, gemini_results, "gemini", OUTPUT_GEMINI)
                except Exception as e:
                    print(f"  ❌ [Gemini] Unexpected error: {e}")
                    gemini_results.append({
                        "Print ID": print_id, "gemini_Theme": "ERROR",
                        "gemini_Motifs": str(e)[:200], "gemini_Style": ""
                    })
        else:
            gemini_results.append({
                "Print ID": print_id, "gemini_Theme": "SKIPPED",
                "gemini_Motifs": "", "gemini_Style": ""
            })

        # ── If BOTH APIs are stopped, break entirely ──
        if openai_stopped and gemini_stopped:
            print("\n🛑 BOTH APIs exhausted/unavailable. Breaking process.")
            # Fill remaining rows with SKIPPED
            remaining = total - seq
            for _ in range(remaining):
                filler_pid = ""
                openai_results.append({
                    "Print ID": filler_pid, "openai_Theme": "SKIPPED",
                    "openai_Motifs": "", "openai_Style": ""
                })
                gemini_results.append({
                    "Print ID": filler_pid, "gemini_Theme": "SKIPPED",
                    "gemini_Motifs": "", "gemini_Style": ""
                })
            break

        # ── Periodic save every 25 images ──
        if seq % 25 == 0:
            print(f"\n💾 Auto-saving intermediary results at {seq}/{total}...")
            _save_intermediary(master_df, openai_results, "openai", OUTPUT_OPENAI)
            _save_intermediary(master_df, gemini_results, "gemini", OUTPUT_GEMINI)

    # ── Final save ──
    print("\n" + "=" * 60)
    print("  Saving final results")
    print("=" * 60)

    _save_intermediary(master_df, openai_results, "openai", OUTPUT_OPENAI)
    _save_intermediary(master_df, gemini_results, "gemini", OUTPUT_GEMINI)
    _save_final(master_df, openai_results, gemini_results)

    print(f"\n🎉 Done! Final output: {OUTPUT_FINAL}")

def _save_intermediary(master_df, results_list, prefix, filepath):
    """Save intermediary results for one provider, appending to existing file."""
    df_res = pd.DataFrame(results_list)
    if df_res.empty:
        print(f"  ℹ️  No new results to save for {prefix}")
        return

    # Build the new rows: merge with master table human columns where available
    new_rows = []
    for _, row in df_res.iterrows():
        pid = row.get(f"{prefix}_PrintID", row.get("Print ID", ""))
        master_match = master_df[master_df["Print ID"] == pid]
        if not master_match.empty:
            human_theme = master_match.iloc[0].get("human_Theme", "")
            human_motifs = master_match.iloc[0].get("human_Motifs", "")
            human_style = master_match.iloc[0].get("human_Style", "")
        else:
            # Image not in master table — leave human columns empty
            human_theme = ""
            human_motifs = ""
            human_style = ""
        new_rows.append({
            "Print ID": pid,
            "human_Theme": human_theme,
            "human_Motifs": human_motifs,
            "human_Style": human_style,
            f"{prefix}_Theme": row.get(f"{prefix}_Theme", ""),
            f"{prefix}_Motifs": row.get(f"{prefix}_Motifs", ""),
            f"{prefix}_Style": row.get(f"{prefix}_Style", ""),
        })

    df_new = pd.DataFrame(new_rows)

    # If file exists, load it and append (avoiding duplicates by Print ID)
    if os.path.exists(filepath):
        df_existing = pd.read_csv(filepath)
        existing_ids = set(df_existing["Print ID"].astype(str))
        df_to_append = df_new[~df_new["Print ID"].astype(str).isin(existing_ids)]
        out = pd.concat([df_existing, df_to_append], ignore_index=True)
    else:
        out = df_new

    out.to_csv(filepath, index=False)
    print(f"  💾 Saved {filepath} ({len(out)} rows)")


def _save_final(master_df, openai_results, gemini_results):
    """Merge everything into the final output CSV, appending to existing file."""
    df_openai = pd.DataFrame(openai_results)
    df_gemini = pd.DataFrame(gemini_results)

    if df_openai.empty and df_gemini.empty:
        print("  ℹ️  No new results to save.")
        return

    # Collect all Print IDs from both result sets
    all_pids = set()
    if not df_openai.empty and "Print ID" in df_openai.columns:
        all_pids.update(df_openai["Print ID"].astype(str))
    if not df_gemini.empty and "Print ID" in df_gemini.columns:
        all_pids.update(df_gemini["Print ID"].astype(str))

    new_rows = []
    for pid in sorted(all_pids):
        master_match = master_df[master_df["Print ID"].astype(str) == pid]
        if not master_match.empty:
            human_theme = master_match.iloc[0].get("human_Theme", "")
            human_motifs = master_match.iloc[0].get("human_Motifs", "")
            human_style = master_match.iloc[0].get("human_Style", "")
        else:
            human_theme = ""
            human_motifs = ""
            human_style = ""

        row_data = {
            "Print ID": pid,
            "human_Theme": human_theme,
            "human_Motifs": human_motifs,
            "human_Style": human_style,
        }

        # Get OpenAI results for this Print ID
        oi_match = df_openai[df_openai["Print ID"].astype(str) == pid] if not df_openai.empty and "Print ID" in df_openai.columns else pd.DataFrame()
        for col in ["openai_Theme", "openai_Motifs", "openai_Style"]:
            row_data[col] = oi_match.iloc[0].get(col, "") if not oi_match.empty else ""

        # Get Gemini results for this Print ID
        gm_match = df_gemini[df_gemini["Print ID"].astype(str) == pid] if not df_gemini.empty and "Print ID" in df_gemini.columns else pd.DataFrame()
        for col in ["gemini_Theme", "gemini_Motifs", "gemini_Style"]:
            row_data[col] = gm_match.iloc[0].get(col, "") if not gm_match.empty else ""

        new_rows.append(row_data)

    df_new = pd.DataFrame(new_rows)

    final_cols = [
        "Print ID", "human_Theme", "human_Motifs", "human_Style",
        "openai_Theme", "openai_Motifs", "openai_Style",
        "gemini_Theme", "gemini_Motifs", "gemini_Style"
    ]
    for c in final_cols:
        if c not in df_new.columns:
            df_new[c] = ""
    df_new = df_new[final_cols]

    # If file exists, load and append (avoiding duplicates)
    if os.path.exists(OUTPUT_FINAL):
        df_existing = pd.read_csv(OUTPUT_FINAL)
        existing_ids = set(df_existing["Print ID"].astype(str))
        df_to_append = df_new[~df_new["Print ID"].astype(str).isin(existing_ids)]
        out = pd.concat([df_existing, df_to_append], ignore_index=True)
    else:
        out = df_new

    out = out[final_cols]
    out.to_csv(OUTPUT_FINAL, index=False)
    print(f"  💾 Saved {OUTPUT_FINAL} ({len(out)} rows)")


# ============================================================
# 🏁  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    run_batch()