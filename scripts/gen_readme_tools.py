"""Regenerate the README's tool section from the DESCRIPTION fields in code (single source of truth)."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.tools import REGISTRY  # noqa: E402

readme = Path(__file__).resolve().parent.parent / "README.md"
blocks = []
for name, tool in REGISTRY.items():
    args = ", ".join(tool.input_model.model_fields)
    lines = "\n".join(f"- {line}" for line in tool.description.strip().splitlines())
    blocks.append(f"#### `{name}({args})`  · retry policy `{tool.retry_policy}`\n{lines}\n")
section = "<!-- TOOLS:START -->\n" + "\n".join(blocks) + "<!-- TOOLS:END -->"
text = re.sub(r"<!-- TOOLS:START -->.*?<!-- TOOLS:END -->", section, readme.read_text(), flags=re.S)
readme.write_text(text)
print(f"README updated with {len(blocks)} tool descriptions")
