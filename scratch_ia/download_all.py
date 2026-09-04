# -*- coding: utf-8 -*-
import urllib.request, urllib.parse, os, re, csv, sys, io
sys.path.insert(0, os.path.dirname(__file__))
from plan import PLAN

DEST_ROOT = r"C:\Users\Gaming PC\Desktop\cc_licensed_music"
MANIFEST = os.path.join(DEST_ROOT, "manifest.csv")

def slugify(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s

# load existing filenames from manifest (dedupe)
existing_filenames = set()
if os.path.exists(MANIFEST):
    with open(MANIFEST, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row:
                existing_filenames.add(row[0])

results = []  # (filename, bucket, artist, title, license, source_url)
used_filenames = set(existing_filenames)

for identifier, file_name, bucket, artist, title, license_str in PLAN:
    base_slug = f"{slugify(artist)}-{slugify(title)}"
    fname = base_slug + ".mp3"
    n = 2
    while fname in used_filenames:
        fname = f"{base_slug}-{n}.mp3"
        n += 1

    dest_dir = os.path.join(DEST_ROOT, bucket)
    dest_path = os.path.join(dest_dir, fname)

    enc_file = urllib.parse.quote(file_name)
    url = f"https://archive.org/download/{identifier}/{enc_file}"
    source_url = f"https://archive.org/details/{identifier}"

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 research-script'})
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, "wb") as out:
            data = resp.read()
            out.write(data)
        size = os.path.getsize(dest_path)
        if size < 20000:
            print(f"FAILED (too small, {size} bytes): {fname}  [{url}]")
            os.remove(dest_path)
            continue
        used_filenames.add(fname)
        results.append((fname, bucket, artist, title, license_str, source_url))
        print(f"OK ({size} bytes): {fname}")
    except Exception as e:
        print(f"FAILED ({e}): {fname}  [{url}]")
        if os.path.exists(dest_path):
            os.remove(dest_path)

# write results to a scratch csv for the final merge step
out_csv = os.path.join(os.path.dirname(__file__), "new_rows.csv")
with open(out_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    for row in results:
        writer.writerow(row)

print(f"\nTotal downloaded: {len(results)}")
