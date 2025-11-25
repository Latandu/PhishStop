import urllib.request
from pathlib import Path

def download_file(url, destination):
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, destination)
    except Exception as e:
        exit(e)


for i in range(2015, 2025):
    base_dir = Path("data/mbox_final")
    url = f"https://monkey.org/~jose/phishing/phishing-{i}"
    destination = base_dir / f"phishing-{i}.mbox"
    download_file(url, destination)

