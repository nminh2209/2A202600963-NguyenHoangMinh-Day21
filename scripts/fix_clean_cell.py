import json
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "notebooks" / "Lab21_LoRA_Finetuning_T4.ipynb"
nb = json.load(open(p, encoding="utf-8"))

NEW_CLEAN = r'''# Clean & deduplicate (rubric: dedup, min output tokens, filter templates)
MIN_OUTPUT_TOKENS = 10
_cols = raw.column_names
_c_instr = next((c for c in ["instruction", "instruction_vi", "prompt", "question"] if c in _cols), None)
_c_input = next((c for c in ["input", "input_vi", "context"] if c in _cols), None)
_c_out = next((c for c in ["output", "output_vi", "response", "answer"] if c in _cols), None)

def clean_dataset(dataset, ic, inc, oc):
    from datasets import Dataset
    seen = set()
    kept = []
    stats = {"dup": 0, "short": 0, "template": 0}
    for ex in dataset:
        instr = ex.get(ic, "")
        inp = (ex.get(inc, "") if inc else "") or ""
        out = ex.get(oc, "") or ""
        key = (instr.strip(), inp.strip(), out.strip())
        if key in seen:
            stats["dup"] += 1
            continue
        seen.add(key)
        if len(out.split()) < MIN_OUTPUT_TOKENS:
            stats["short"] += 1
            continue
        if out.strip().startswith("###") or "Write your response" in out:
            stats["template"] += 1
            continue
        kept.append(ex)
    print(f"✓ Clean: {len(kept)}/{len(dataset)} kept | dup={stats['dup']} short={stats['short']} template={stats['template']}")
    return Dataset.from_list(kept) if kept else dataset

raw = clean_dataset(raw, _c_instr, _c_input, _c_out)
'''

for i, c in enumerate(nb["cells"]):
    if "Clean & deduplicate" in "".join(c["source"]):
        nb["cells"].pop(i)
        break

for j, c in enumerate(nb["cells"]):
    if "Auto-detect column names" in "".join(c["source"]):
        nb["cells"].insert(
            j,
            {
                "cell_type": "code",
                "metadata": {"id": "cell-clean"},
                "source": [line + "\n" for line in NEW_CLEAN.split("\n")],
                "outputs": [],
            },
        )
        print(f"Inserted clean before format at index {j}")
        break

json.dump(nb, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
