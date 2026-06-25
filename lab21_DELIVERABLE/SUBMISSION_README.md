# Lab 21 Submission Package

Cấu trúc này khớp **Option A** trong `Lab21_Rubric_and_Format.md`.

```
lab21_<MSSV>/
├── REPORT.md
├── notebook.ipynb          ← copy từ notebooks/Lab21_LoRA_Finetuning_T4.ipynb (clear outputs)
├── adapters/
│   └── r16/                ← best rank only (sau khi train trên Colab)
├── results/
│   ├── rank_experiment_summary.csv
│   ├── qualitative_comparison.csv
│   └── loss_curve.png
└── requirements.txt
```

## Cách hoàn thành

1. Upload `notebooks/Lab21_LoRA_Finetuning_T4.ipynb` lên **Google Colab** (GPU T4).
2. Điền `STUDENT_NAME`, `STUDENT_ID` trong cell config.
3. **Run All** (~60 phút).
4. Download `OUTPUT_DIR` từ Colab → copy `adapters/r16/` vào đây.
5. Đổi tên folder thành `lab21_<MSSV>_<HoTen>` và zip.

## Bonus points

| Bonus | Cách đạt |
|-------|----------|
| +5 Option B | `PUSH_TO_HUB=True` + HF token |
| +10 Stretch | `USE_WANDB=True`, `RUN_STRETCH_ALL_LAYERS=True`, hoặc custom dataset ≥200 |
