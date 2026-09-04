"""Sanity-check the shipped page without a JS runtime.

Primary target is `index.html` -- that is the file the Cloudflare Worker serves
verbatim AND the file the local server reads from disk, so it is what actually
ships. (`INDEX_HTML` inside 09_api_server.py is only a fallback for when
index.html is missing; it is checked for drift, not correctness.)

Checks:
  1. no duplicate id attributes
  2. every getElementById('x') in the JS has a matching id="x" in the HTML
  3. HTML tags balance -- an unclosed <div> silently destroys the grid layout
  4. brackets balance inside <script> (string/comment/regex aware)
  5. no unrendered __PLACEHOLDER__ tokens (the Worker does no substitution)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEPLOYED = ROOT / "index.html"
SERVER_FILE = ROOT / "scripts" / "09_api_server.py"

VOID = {"br", "img", "input", "meta", "link", "hr", "source", "path", "circle",
        "line", "polygon", "text", "use", "stop", "rect", "ellipse", "col", "area"}

failures = []


def fail(msg):
    print(f"  FAIL: {msg}")
    failures.append(msg)


def ok(msg):
    print(f"  ok: {msg}")


def scripts_of(page):
    return "\n".join(re.findall(r"<script[^>]*>(.*?)</script>", page, re.S))


def check_ids(page, script):
    all_ids = re.findall(r'\sid="([^"]+)"', page)
    # ids built inside JS template literals aren't real page ids
    all_ids = [i for i in all_ids if "${" not in i]
    dupes = sorted({i for i in all_ids if all_ids.count(i) > 1})
    if dupes:
        fail(f"duplicate ids: {dupes}")
    else:
        ok("no duplicate ids")

    js_ids = set(re.findall(r"getElementById\(['\"]([^'\"]+)['\"]\)", script))
    js_ids = {i for i in js_ids if "${" not in i}
    missing = sorted(js_ids - set(all_ids))
    if missing:
        fail(f"JS references {len(missing)} element(s) that do not exist: {missing}")
    else:
        ok(f"every getElementById target exists ({len(js_ids)} lookups, {len(set(all_ids))} ids)")


def check_handlers(page, script):
    """An onclick="foo()" with no foo defined is a button that silently does
    nothing -- the kind of thing only found by clicking it in front of people."""
    html = re.sub(r"<script[^>]*>.*?</script>", "", page, flags=re.S)
    referenced = set()
    for m in re.finditer(r'on(?:click|change|input|submit)="([^"]+)"', html):
        # (?<![.\w]) skips method calls like el.click() / document.getElementById()
        referenced |= set(re.findall(r'(?<![.\w])([A-Za-z_$][\w$]*)\s*\(', m.group(1)))

    defined = set(re.findall(r"function\s+([A-Za-z_$][\w$]*)\s*\(", script))
    defined |= set(re.findall(r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function|\()", script))
    defined |= {"alert", "confirm", "open", "parseInt", "parseFloat", "encodeURIComponent",
                "Number", "String", "Boolean", "event"}

    missing = sorted(referenced - defined)
    if missing:
        fail(f"inline handler(s) call undefined function(s) -- those controls do nothing: {missing}")
    else:
        ok(f"all {len(referenced)} inline handlers resolve to defined functions")


def check_tags(page):
    # bound to <body>...</body>: starting at <body> without stopping at </body>
    # leaves </html> closing a tag that was never pushed onto the stack
    body_start = page.index("<body>")
    body_end = page.rindex("</body>") if "</body>" in page else len(page)
    # only structural markup, not the JS that builds markup in strings
    body = re.sub(r"<script[^>]*>.*?</script>", "", page[body_start:body_end], flags=re.S)
    stack = []
    for m in re.finditer(r"<(/?)([a-zA-Z][\w-]*)([^>]*)>", body):
        closing, tag, attrs = m.group(1), m.group(2).lower(), m.group(3)
        if tag in VOID or attrs.rstrip().endswith("/"):
            continue
        if closing:
            if not stack or stack[-1][0] != tag:
                line = body[:m.start()].count("\n") + 1
                expected = stack[-1][0] if stack else "nothing"
                fail(f"tag mismatch at body line ~{line}: </{tag}> but expected </{expected}>")
                return
            stack.pop()
        else:
            stack.append((tag, m.start()))
    leftover = [t for t, _ in stack if t != "body"]
    if leftover:
        fail(f"unclosed tags: {leftover}")
    else:
        ok("html tags balanced")


def check_brackets(script):
    pairs = {"}": "{", ")": "(", "]": "["}
    stack, i, in_str = [], 0, None
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
        elif c == "/" and i + 1 < len(script) and script[i + 1] == "/":
            nl = script.find("\n", i)
            if nl == -1:
                break
            i = nl
            continue
        elif c == "/" and i + 1 < len(script) and script[i + 1] == "*":
            i = script.find("*/", i) + 2
            continue
        elif c == "/" and i + 1 < len(script) and (
            (lambda p: p == "" or p in "=(,:[!&|?{};+")(
                next((ch for ch in reversed(script[:i]) if not ch.isspace()), "")
            )
        ):
            # regex literal: skip to the closing '/', respecting [...] classes
            j, in_class = i + 1, False
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
        elif c in "{([":
            stack.append((c, script.count("\n", 0, i) + 1))
        elif c in "})]":
            if not stack or stack[-1][0] != pairs[c]:
                fail(f"unbalanced {c!r} at script line {script.count(chr(10), 0, i) + 1}")
                return
            stack.pop()
        i += 1
    if stack:
        fail("unclosed openers at script lines: " + ", ".join(f"{c!r}@{ln}" for c, ln in stack[:5]))
    else:
        ok("brackets balanced")


def check_deployable(page):
    left = [t for t in ("__APP_VERSION__", "__FORMAT_OPTIONS__") if t in page]
    if left:
        # the Worker serves this verbatim: placeholders reach real users
        fail(f"unrendered placeholders would ship to production: {left}")
    elif not re.search(r'<option value="(WAV|FLAC|OGG|MP3)"', page):
        fail("no format <option> elements -- the Format dropdown would render empty")
    else:
        ok("rendered and deployable")


def main():
    # optional path override, so the checks can be run against any page
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEPLOYED
    if not target.is_file():
        sys.exit(f"missing {target}")
    page = target.read_text(encoding="utf-8", errors="replace")
    script = scripts_of(page)

    print(f"index.html ({len(page) / 1024:.0f} KB, {len(script) / 1024:.0f} KB of JS)")
    check_ids(page, script)
    check_handlers(page, script)
    check_tags(page)
    check_brackets(script)
    check_deployable(page)

    # The inline copy is only a fallback; warn when it has drifted far enough
    # that serving it would give a noticeably different app.
    if SERVER_FILE.is_file():
        src = SERVER_FILE.read_text(encoding="utf-8", errors="replace")
        marker = 'INDEX_HTML = """'
        if marker in src:
            s = src.index(marker) + len(marker)
            inline = src[s:src.index('"""', s)]
            drift = abs(len(inline) - len(page))
            print(f"  note: inline fallback differs from index.html by {drift} chars"
                  f"{' (stale)' if drift > 2000 else ''}")

    print("PASS" if not failures else f"FAILED ({len(failures)} issue(s))")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
