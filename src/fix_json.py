import json, re
from pathlib import Path

p = Path("data/llamafactory/metal_defect_test.json")
raw = p.read_text(encoding="utf-8")

# 策略1：尝试 json5 风格容错解析（需 pip install json5）
try:
    import json5
    data = json5.loads(raw)
    print("json5 容错解析成功，样本数:", len(data))
except Exception as e:
    print("json5 也失败:", e)
    raise

# 写回标准合法 JSON（ensure_ascii=False 保留中文）
out = Path("data/llamafactory/metal_defect_test_fixed.json")
out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print("已写出:", out)