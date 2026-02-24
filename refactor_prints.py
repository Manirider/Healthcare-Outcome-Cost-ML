import os
import re

files_to_check = []
for root, _, files in os.walk("src"):
    for f in files:
        if f.endswith(".py") and f != "logger.py":
            files_to_check.append(os.path.join(root, f))
files_to_check.append("run_all.py")

import_stmt = "from src.logger import get_logger\nlogger = get_logger(__name__)"

for fp in files_to_check:
    print(f"Modifying {fp}...")
    with open(fp, "r", encoding="utf-8") as f:
        content = f.read()

    if "logger = get_logger" in content:
        continue

    new_content = re.sub(r'(?<!\.)\bprint\(', 'logger.info(', content)

    lines = new_content.split("\n")
    last_import_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            last_import_idx = i

    if last_import_idx != -1:
        lines.insert(last_import_idx + 1, import_stmt)
    else:
        lines.insert(0, import_stmt)

    with open(fp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

print("Refactoring complete.")

