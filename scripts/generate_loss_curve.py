"""Generate loss_curve.png from r=16 training log (Colab run)."""
import matplotlib.pyplot as plt
from pathlib import Path

steps = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
losses = [1.614305, 1.573560, 1.606689, 1.555441, 1.479080, 1.416185,
          1.496181, 1.480136, 1.380218, 1.388354, 1.424118, 1.413686, 1.394213]

out = Path(__file__).resolve().parents[1] / "lab21_DELIVERABLE" / "results" / "loss_curve.png"
out.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 4))
plt.plot(steps, losses, label="train", color="#0E2A52", linewidth=2)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss Curve — r=16 (T4, eval-during-training off)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out, dpi=150)
print(f"Saved {out}")
