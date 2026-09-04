"""Sanity-check the inline HTML/JS in 09_api_server.py without a JS runtime:
  1. every getElementById('x') in the JS has a matching id="x" in the HTML
  2. brackets balance inside the <script> block
  3. no leftover placeholder tokens after substitution
"""
import re
import sys
from pathlib import Path

src = (Path(__file__).resolve().parent / "09_api_server.py").read_text(encoding="utf-8")

start = src.index('INDEX_HTML = """') + len('INDEX_HTML = """')
end = src.index('"""', start)
page = src[start:end]

script = page[page.index("<script>") + len("<script>"): page.index("</script>")]

all_id_attrs = re.findall(r'\sid="([^"]+)"', page)
dupes = sorted({i for i in all_id_attrs if all_id_attrs.count(i) > 1})
if dupes:
    print("DUPLICATE IDS:", dupes)
else:
    print("OK: no duplicate ids")

html_ids = set(all_id_attrs)
js_ids = set(re.findall(r"getElementById\('([^']+)'\)", script))
js_ids |= set(re.findall(r'getElementById\("([^"]+)"\)', script))
# ids built dynamically inside template literals are skipped
js_ids = {i for i in js_ids if "${" not in i}

missing = sorted(js_ids - html_ids)
print(f"HTML ids: {len(html_ids)} | JS lookups: {len(js_ids)}")
if missing:
    print("MISSING IDS (JS references an element that does not exist):")
    for m in missing:
        print("  -", m)
else:
    print("OK: every getElementById target exists")

# HTML tag nesting: an unclosed <div> silently destroys the grid layout
body = page[page.index("<body>"): page.index("<script>")]
VOID = {"br", "img", "input", "meta", "link", "hr", "source", "path", "circle", "line", "polygon", "text"}
tag_stack, tag_bad = [], False
for m in re.finditer(r"<(/?)(\w+)([^>]*)>", body):
    closing, tag, attrs = m.group(1), m.group(2).lower(), m.group(3)
    if tag in VOID or attrs.rstrip().endswith("/"):
        continue
    if closing:
        if not tag_stack or tag_stack[-1] != tag:
            print(f"HTML MISMATCH: </{tag}> at offset {m.start()}, expected </{tag_stack[-1] if tag_stack else '?'}>")
            tag_bad = True
            break
        tag_stack.pop()
    else:
        tag_stack.append(tag)
leftover_tags = [t for t in tag_stack if t != "body"]
if not tag_bad:
    print("OK: html tags balanced" if not leftover_tags else f"UNCLOSED TAGS: {leftover_tags}")

pairs = {"}": "{", ")": "(", "]": "["}
stack, bad = [], False
in_str = None
i = 0
while i < len(script):
    c = script[i]
    if in_str:
        if c == "\\":
            i += 2
            continue
        if c == in_str:
            in_str = None
    elif c in "\"'`":
        in_str = c
    elif c == "/" and i + 1 < len(script) and script[i + 1] not in "/*" and (
        # a '/' in operand position starts a regex literal, not a division
        (lambda p: p == "" or p in "=(,:[!&|?{};+")(
            next((ch for ch in reversed(script[:i]) if not ch.isspace()), "")
        )
    ):
        j = i + 1
        in_class = False
        while j < len(script):
            if script[j] == "\\":
                j += 2
                continue
            if script[j] == "[":
                in_class = True
            elif script[j] == "]":
                in_class = False
            elif script[j] == "/" and not in_class:
                break
            elif script[j] == "\n":
                break
            j += 1
        i = j + 1
        continue
    elif c == "/" and i + 1 < len(script) and script[i + 1] == "/":
        i = script.find("\n", i)
        if i == -1:
            break
        continue
    elif c == "/" and i + 1 < len(script) and script[i + 1] == "*":
        i = script.find("*/", i) + 2
        continue
    elif c in "{([":
        stack.append((c, script.count("\n", 0, i) + 1))
    elif c in "})]":
        if not stack or stack[-1][0] != pairs[c]:
            line = script.count("\n", 0, i) + 1
            ctx = script.splitlines()[line - 1] if line <= len(script.splitlines()) else ""
            print(f"UNBALANCED {c!r} at script line {line}: {ctx.strip()[:90]}")
            bad = True
            break
        stack.pop()
    i += 1

if not bad:
    if not stack:
        print("OK: brackets balanced")
    else:
        lines = script.splitlines()
        print("UNCLOSED openers:")
        for ch, ln in stack:
            ctx = lines[ln - 1].strip()[:90] if ln <= len(lines) else ""
            print(f"  {ch!r} opened at script line {ln}: {ctx}")

leftover = [t for t in ("__APP_VERSION__", "__FORMAT_OPTIONS__") if t in page]
print("Placeholders present in template (expected):", leftover)

# worker.js serves index.html verbatim -- no substitution happens at the edge.
# An unrendered placeholder there means production ships a literal
# "v__APP_VERSION__" footer and a Format <select> with no <option> elements,
# which renders as an empty, unusable dropdown. This shipped once already.
deployed_path = Path(__file__).resolve().parent.parent / "index.html"
deployed_bad = []
if deployed_path.is_file():
    deployed = deployed_path.read_text(encoding="utf-8", errors="replace")
    deployed_bad = [t for t in ("__APP_VERSION__", "__FORMAT_OPTIONS__") if t in deployed]
    if deployed_bad:
        print(f"DEPLOY BLOCKER: index.html has unrendered placeholders {deployed_bad}")
    elif not re.search(r'<option value="(WAV|FLAC|OGG|MP3)"', deployed):
        deployed_bad = ["no format <option> elements"]
        print("DEPLOY BLOCKER: index.html has no format <option> elements")
    else:
        print("OK: index.html is rendered and deployable")
else:
    print("note: index.html not found next to scripts/ (skipping deploy check)")

sys.exit(1 if missing or bad or stack or dupes or tag_bad or leftover_tags or deployed_bad else 0)
