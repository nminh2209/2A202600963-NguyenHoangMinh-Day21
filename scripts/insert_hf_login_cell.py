import json
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "notebooks" / "Lab21_LoRA_Finetuning_T4.ipynb"
nb = json.load(open(p, encoding="utf-8"))

HF_CELL = '''# HuggingFace Hub login (Option B · +5 bonus)
# 1) Create account: https://huggingface.co/join
# 2) Create token:   https://huggingface.co/settings/tokens → New token → Role: Write
# 3) Colab: add secret HF_TOKEN in 🔑 Secrets (left sidebar)
#    Local: set env HF_TOKEN=hf_... before running

def setup_hf_hub():
    if not PUSH_TO_HUB:
        print("ℹ PUSH_TO_HUB=False — skip HF login")
        return
    if HF_USERNAME == "YOUR_HF_USERNAME":
        raise ValueError("Set HF_USERNAME in the config cell (your huggingface.co profile name)")
    from huggingface_hub import login
    import os
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        except Exception:
            pass
    if not token:
        from getpass import getpass
        token = getpass("Paste HF write token (hf_...): ")
    login(token=token)
    print(f"✓ Logged in | will push to: https://huggingface.co/{HUB_REPO_ID}")

setup_hf_hub()
'''

# Skip if already inserted
if not any("setup_hf_hub" in "".join(c.get("source", [])) for c in nb["cells"]):
    for i, c in enumerate(nb["cells"]):
        if "cell-config" in c.get("metadata", {}).get("id", "") or "HUB_REPO_ID = f" in "".join(c.get("source", [])):
            nb["cells"].insert(
                i + 1,
                {
                    "cell_type": "code",
                    "metadata": {"id": "cell-hf-login"},
                    "source": [line + "\n" for line in HF_CELL.split("\n")],
                    "outputs": [],
                },
            )
            print("Inserted HF login cell")
            break

# Simplify push cell — login already done at start
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if "Option B bonus (+5): Push best adapter" in src:
        c["source"] = [
            "# Option B bonus (+5): Push best adapter to HuggingFace Hub\n",
            "if PUSH_TO_HUB:\n",
            "    ft_model.push_to_hub(HUB_REPO_ID)\n",
            "    tok_for_eval.push_to_hub(HUB_REPO_ID)\n",
            "    links_path = os.path.join(OUTPUT_DIR, \"LINKS.md\")\n",
            "    with open(links_path, \"w\", encoding=\"utf-8\") as f:\n",
            "        f.write(f\"# Lab 21 Links\\n\\n\")\n",
            "        f.write(f\"**Student**: {STUDENT_NAME} — {STUDENT_ID}\\n\\n\")\n",
            "        f.write(f\"- HF Hub adapter: https://huggingface.co/{HUB_REPO_ID}\\n\")\n",
            "    print(f\"✓ Adapter pushed: https://huggingface.co/{HUB_REPO_ID}\")\n",
            "    print(f\"✓ LINKS.md saved: {links_path}\")\n",
            "else:\n",
            '    print("ℹ PUSH_TO_HUB=False — skip Hub upload")\n',
        ]
        print("Updated push cell")
        break

json.dump(nb, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
