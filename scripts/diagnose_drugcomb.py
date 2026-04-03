"""DrugComb connection diagnosis."""
import requests, socket, ssl

print('=== DrugComb Connection Diagnosis ===')

# 1. DNS check
host = 'drugcomb.fimm.fi'
print(f'\n1. DNS Resolution: {host}')
try:
    ips = socket.getaddrinfo(host, 443)
    for ip in ips[:3]:
        print(f'   -> {ip[4][0]}:{ip[4][1]}')
except Exception as e:
    print(f'   DNS FAILED: {e}')

# 2. TCP connection
print(f'\n2. TCP Connection to {host}:443')
try:
    sock = socket.create_connection((host, 443), timeout=10)
    print(f'   -> Connected OK')
    sock.close()
except Exception as e:
    print(f'   -> FAILED: {e}')

# 3. HTTP
print(f'\n3. HTTP GET https://{host}/')
try:
    resp = requests.get(f'https://{host}/', timeout=15)
    print(f'   -> Status: {resp.status_code}')
    print(f'   -> Size: {len(resp.content)} bytes')
except Exception as e:
    err_msg = str(e).split('(Caused')[0][:120]
    print(f'   -> FAILED: {err_msg}')

# 4. Alternative endpoints
print(f'\n4. Trying alternatives...')
alts = [
    'https://drugcomb.fimm.fi/api/drugs',
    'https://drugcomb.fimm.fi/api/cellines',
    'https://api.drugcomb.org/',
    'https://drugcomb.org/',
]
for url in alts:
    try:
        r = requests.get(url, timeout=10)
        print(f'   {url} -> {r.status_code} ({len(r.content)} bytes)')
    except Exception as e:
        err_msg = str(e).split('(Caused')[0][:80]
        print(f'   {url} -> FAIL: {err_msg}')

# 5. Check Zenodo for DrugComb data
print(f'\n5. Searching Zenodo for DrugComb data...')
try:
    r = requests.get('https://zenodo.org/api/records?q=drugcomb+synergy&size=5', timeout=15)
    if r.status_code == 200:
        hits = r.json().get('hits', {}).get('hits', [])
        for h in hits[:5]:
            title = h.get('metadata', {}).get('title', '')
            doi = h.get('doi', '')
            files = h.get('files', [])
            print(f'   [{doi}] {title[:70]}')
            for f in files[:2]:
                key = f.get('key', '')
                size_mb = f.get('size', 0) / 1e6
                dl = f.get('links', {}).get('self', '')
                print(f'     -> {key} ({size_mb:.1f} MB) {dl[:80]}')
except Exception as e:
    print(f'   Zenodo error: {e}')

# 6. Check SynergyFinder/other drug combo databases
print(f'\n6. Alternative synergy databases...')
alt_dbs = [
    ('https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178318', 'GEO drug combo'),
    ('https://synergyfinder.fimm.fi/', 'SynergyFinder'),
]
for url, name in alt_dbs:
    try:
        r = requests.get(url, timeout=10)
        print(f'   {name}: {r.status_code}')
    except Exception as e:
        err_msg = str(e).split('(Caused')[0][:60]
        print(f'   {name}: FAIL - {err_msg}')
