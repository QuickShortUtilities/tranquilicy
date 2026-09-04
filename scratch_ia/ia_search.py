import urllib.request, urllib.parse, json, sys, re

def search(query, rows=75):
    params = {
        'q': query,
        'fl[]': ['identifier','title','creator','licenseurl','collection','mediatype','downloads'],
        'rows': str(rows),
        'output': 'json',
        'sort[]': 'downloads desc',
    }
    qs = urllib.parse.urlencode(params, doseq=True)
    url = f"https://archive.org/advancedsearch.php?{qs}"
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0 research-script'})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    return data.get('response', {}).get('docs', [])

GOOD_LICENSE_RE = re.compile(r'(licenses/by/[\d.]+|licenses/by-sa/[\d.]+|publicdomain/zero/)', re.I)
BAD_LICENSE_RE = re.compile(r'(nc|nd)[/-]|-nc|-nd', re.I)

def is_good_license(url):
    if not url:
        return False
    u = url.lower()
    if 'by-nc' in u or 'by-nd' in u or '/nc/' in u or '/nd/' in u:
        return False
    return bool(GOOD_LICENSE_RE.search(u))

if __name__ == '__main__':
    query = sys.argv[1]
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 75
    docs = search(query, rows)
    good = [d for d in docs if is_good_license(d.get('licenseurl',''))]
    print(f"TOTAL={len(docs)} GOOD_LICENSE={len(good)}")
    for d in good:
        print(json.dumps({
            'identifier': d.get('identifier'),
            'title': d.get('title'),
            'creator': d.get('creator'),
            'licenseurl': d.get('licenseurl'),
            'collection': d.get('collection'),
        }))
