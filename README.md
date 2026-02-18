# 🎨 AI Design Tagger

Automated image tagging pipeline. Uses **OpenAI** and **Google Gemini** vision models to classify print-on-demand shirt artwork into structured metadata (Theme, Motifs, Style) compatible with Adobe Creative Cloud.

---

## Overview

The pipeline reads shirt design images, sends them through both OpenAI and Gemini vision APIs with a few-shot prompting strategy built from human-tagged examples, and outputs a merged CSV with side-by-side tags from each provider. This dual-model approach lets you compare outputs and choose the best tags per design.

### Key Features

- **Dual-model tagging** — Runs both OpenAI (GPT) and Gemini in parallel, producing side-by-side results for comparison
- **Few-shot prompting** — Uses ~19 human-tagged example designs to guide the AI toward brand-consistent outputs
- **Schema-driven** — Loads allowed Theme/Motifs/Style values from an editable `tagging_schema.csv` or `.xlsx` file
- **Crash-resilient** — Caches intermediary results every 25 images; resumes from where it left off on restart
- **Credit-aware** — Detects API quota/billing errors and gracefully stops that provider while continuing with the other
- **Auto-generates master table** — If no `master_table.csv` exists, builds one from image filenames
- **XMP embedding** — Notebook includes cells for writing tags directly into PNG metadata (Adobe-compatible `iTXt` XMP chunks)
- **HTML Viewer** — Bundled single-file HTML app for browsing tagged images with Grid, List, and Detail views

---

## Project Structure

```
├── tagging_engine.py              # Main CLI script — run this for batch tagging
├── shirt_design_tagger.ipynb      # Jupyter notebook (interactive/exploratory version)
├── image_tag_viewer.html          # Browser-based viewer for tagged results
├── tagging_schema.csv/.xlsx       # Editable schema: allowed Theme, Motifs, Style values
├── master_table.csv/.xlsx         # Input: Print IDs + optional human tags
├── artwork_images/                # Input: shirt design PNGs/JPGs
│   ├── CT1230.png
│   ├── CT1231.png
│   └── ...
├── jns_assets/                    # Few-shot example images (referenced by the prompt)
│   ├── artwork_870.png
│   ├── artwork_871.png
│   └── ...
├── .env                           # API keys (not committed)
└── output/
    ├── openai_tagged_master_table_jns.csv   # Intermediary — OpenAI results
    ├── gemini_tagged_master_table_jns.csv   # Intermediary — Gemini results
    └── tagged_master_table_jns.csv          # Final merged output
```

---

## Setup

### 1. Install Dependencies

```bash
pip install openai google-generativeai pandas python-dotenv openpyxl Pillow
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
```

Both keys are optional — the pipeline will skip whichever provider has no key set.

### 3. Prepare Inputs

| Input | Location | Required |
|---|---|---|
| Shirt design images | `./artwork_images/` | Yes |
| Tagging schema | `tagging_schema.csv` or `.xlsx` | Yes |
| Master table | `master_table.csv` or `.xlsx` | No (auto-generated from filenames) |
| Few-shot example images | `./jns_assets/` | Recommended |

**Tagging schema format** — Three columns with allowed values listed as rows:

```csv
Theme,Motifs,Style
Outdoors,mountain,retro
Mountains,pine tree,vintage
Ski,skiing,line art
...,...,...
```

**Master table format** — At minimum a `Print ID` column. Optionally includes human tags:

```csv
Print ID,Theme,Motifs,Style
CT1230,Ski,"gondola, ski lift, Vail","line art, minimalist"
```

---

## Usage

### CLI (Recommended for batch runs)

```bash
python tagging_engine.py
```

The script will:
1. Load the tagging schema and master table
2. Match each Print ID to an image file in `artwork_images/`
3. Tag each image with OpenAI and Gemini (skipping already-cached results)
4. Auto-save intermediary CSVs every 25 images
5. Output the final merged CSV to `tagged_master_table_jns.csv`

### Jupyter Notebook (Exploratory)

Open `shirt_design_tagger.ipynb` for step-by-step execution — useful for testing single images, adjusting prompts, validating AI vs human tags, and embedding XMP metadata into PNGs.

---

## Output Format

The final `tagged_master_table_jns.csv` has this structure:

| Column | Description |
|---|---|
| `Print ID` | Unique artwork identifier (matches filename) |
| `human_Theme` | Human-assigned theme (if available) |
| `human_Motifs` | Human-assigned motifs (if available) |
| `human_Style` | Human-assigned style (if available) |
| `openai_Theme` | OpenAI-predicted theme |
| `openai_Motifs` | OpenAI-predicted motifs |
| `openai_Style` | OpenAI-predicted style |
| `gemini_Theme` | Gemini-predicted theme |
| `gemini_Motifs` | Gemini-predicted motifs |
| `gemini_Style` | Gemini-predicted style |

---

## Image Tag Viewer

A standalone HTML app (`image_tag_viewer.html`) for browsing results visually. Open it in any browser — no server needed.

### How to use

1. Open `image_tag_viewer.html` in a browser
2. Select your `artwork_images/` folder
3. Select the output CSV (`tagged_master_table_jns.csv`)
4. Browse in three view modes:

| View | Description |
|---|---|
| **Grid** | Thumbnail cards with tag pills; adjustable card size |
| **List** | Horizontal rows with full OpenAI + Gemini tags side by side |
| **Detail** | Single image with full tag panels; arrow-key navigation |

Click any card in Grid/List to open a modal with full details. Search filters across Print ID and all tag fields.

---

## Configuration

Key settings in `tagging_engine.py`:

```python
IMAGE_FOLDER   = "./artwork_images"
OPENAI_MODEL   = "gpt-5-mini-2025-08-07"   # Change to gpt-4o, gpt-4.1, etc.
GEMINI_MODEL   = "gemini-2.5-flash"
API_DELAY      = 1.0                         # Seconds between API calls
```

---

## Prompt Design

The system prompt enforces:

- **Strict JSON output** — No markdown fences, no conversational filler
- **Single theme** — Exactly one primary environment/activity from the schema
- **130-character limit** — Combined motifs + style must stay under 130 chars (Adobe tag limit)
- **Specificity** — Prioritizes concrete nouns ("Grizzly Bear" over "Animal") and transcribes all visible text
- **Brand alignment** — Tuned for client's aesthetic: National Parks, Alpine, Coastal, Western

Few-shot examples (19 human-tagged designs) are sent as image+JSON pairs in each API call to anchor the model's behavior.

---

## Error Handling

- **API credit exhaustion** — Detected automatically via error message heuristics. The exhausted provider is stopped; the other continues. Intermediary results are saved immediately.
- **JSON parse failures** — Recorded as `PARSE_ERROR` with the raw response text preserved in the motifs column.
- **Missing images** — Print IDs without matching image files are logged and skipped.
- **Resume on crash** — Existing intermediary CSVs are loaded on startup; already-tagged Print IDs are skipped.

---

## XMP Metadata Embedding

The notebook includes cells for writing tags directly into PNG files as Adobe-compatible XMP metadata, using a pure-Python approach (no `exiftool` dependency). This writes an `iTXt` chunk with keyword `XML:com.adobe.xmp`, readable by Adobe Creative Cloud applications.

---

## License

Not intended for public distribution.
