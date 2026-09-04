# -*- coding: utf-8 -*-
import csv, os, time

MANIFEST = r"C:\Users\Gaming PC\Desktop\cc_licensed_music\manifest.csv"
NEW_ROWS = r"C:\Users\Gaming PC\chillout-musicgen\scratch_ia\new_rows.csv"

def read_filenames():
    fnames = set()
    with open(MANIFEST, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row:
                fnames.add(row[0])
    return fnames

size_before = os.path.getsize(MANIFEST)
existing = read_filenames()

with open(NEW_ROWS, "r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    new_rows = [row for row in reader]

# filter out any filename that's already present (race-safety / no dup)
to_write = [row for row in new_rows if row[0] not in existing]
skipped = [row for row in new_rows if row[0] in existing]

# re-check size right before writing; if changed, re-read to be safe (append mode is non-destructive anyway)
size_now = os.path.getsize(MANIFEST)
if size_now != size_before:
    print(f"NOTE: manifest size changed between read ({size_before}) and write ({size_now}) - re-checking dedupe set")
    existing2 = read_filenames()
    to_write = [row for row in new_rows if row[0] not in existing2]
    skipped = [row for row in new_rows if row[0] in existing2]

with open(MANIFEST, "a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    for row in to_write:
        writer.writerow(row)

print(f"Appended {len(to_write)} rows. Skipped {len(skipped)} (already present).")
size_after = os.path.getsize(MANIFEST)
print(f"Manifest size: before={size_before} now-before-write={size_now} after={size_after}")
