import urllib.request, json, sys, re

def get_meta(identifier):
    url = f"https://archive.org/metadata/{identifier}"
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0 research-script'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

def is_good_license(url):
    if not url:
        return False
    u = url.lower()
    if 'by-nc' in u or 'by-nd' in u or '/nc/' in u or '/nd/' in u:
        return False
    return bool(re.search(r'(licenses/by/[\d.]+|licenses/by-sa/[\d.]+|publicdomain/zero/)', u))

if __name__ == '__main__':
    identifier = sys.argv[1]
    d = get_meta(identifier)
    m = d.get('metadata', {})
    lic = m.get('licenseurl')
    print(f"IDENTIFIER={identifier}")
    print(f"TITLE={m.get('title')}")
    print(f"CREATOR={m.get('creator') or m.get('artist')}")
    print(f"LICENSEURL={lic}")
    print(f"GOOD_LICENSE={is_good_license(lic)}")
    files = d.get('files', [])
    audio_files = [f for f in files if f.get('format') in ('VBR MP3','MP3','Ogg Vorbis','FLAC','24bit FLAC','WAVE','Flac')]
    print(f"NUM_AUDIO_FILES={len(audio_files)}")
    for f in audio_files[:15]:
        print(f"  FILE name={f.get('name')} format={f.get('format')} length={f.get('length')} size={f.get('size')} license_from_file={f.get('license') if 'license' in f else ''}")
