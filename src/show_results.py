import json

with open("results/benchmark_results.json", encoding="utf-8") as f:
    data = json.load(f)

r = [x for x in data if x["prompt_type"] == "EP"][0]
print(f"Model: {r['model']}")
print(f"Accuracy: {r['accuracy']:.4f}  Macro-F1: {r['f1_macro']:.4f}")
print()
print("Per-class F1:")
print("-" * 50)
for cls, m in r["per_class"].items():
    if isinstance(m, dict) and "f1-score" in m:
        bar = "█" * int(m["f1-score"] * 40)
        print(f"  {cls:25s} F1={m['f1-score']:.3f}  {bar}")