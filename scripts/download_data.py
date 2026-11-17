"""Download the raw UCI HAR and HAPT archives needed by Module 2 (§6-§11) of the notebook.

Usage (from the repository root):
    python scripts/download_data.py                  # both datasets
    python scripts/download_data.py --only uci       # just UCI HAR (needed by §6-§10)
    python scripts/download_data.py --only hapt      # just HAPT (needed by §11)
    python scripts/download_data.py --from-zip uci path/to/downloaded.zip   # use a zip you downloaded manually

Files are unpacked into data/external/ exactly where the notebook expects them. Uses only the Python
standard library. Safe to re-run: datasets that are already in place are skipped.
"""
import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

EXTERNAL = Path(__file__).resolve().parents[1] / "data" / "external"

DATASETS = {
    "uci": {
        "title": "Human Activity Recognition Using Smartphones (UCI dataset 240)",
        "url": "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
        "folder": "human+activity+recognition+using+smartphones",
        # files the notebook reads, relative to data/external/<folder>/
        "required": ["UCI HAR Dataset/train/subject_train.txt", "UCI HAR Dataset/train/X_train.txt",
                     "UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt",
                     "UCI HAR Dataset/test/subject_test.txt", "UCI HAR Dataset/test/X_test.txt",
                     "UCI HAR Dataset/test/Inertial Signals/total_acc_x_test.txt"],
        "anchor": "UCI HAR Dataset",          # directory that must sit directly inside <folder>
    },
    "hapt": {
        "title": "Smartphone-Based Recognition of Human Activities and Postural Transitions (UCI dataset 341)",
        "url": "https://archive.ics.uci.edu/static/public/341/"
               "smartphone+based+recognition+of+human+activities+and+postural+transitions.zip",
        "folder": "smartphone+based+recognition+of+human+activities+and+postural+transitions",
        "required": ["RawData/labels.txt", "Train/X_train.txt", "Train/y_train.txt", "Train/subject_id_train.txt",
                     "Test/X_test.txt", "Test/y_test.txt", "Test/subject_id_test.txt"],
        "anchor": None,                        # RawData/, Train/, Test/ sit directly inside <folder>
    },
}


def missing_files(spec):
    root = EXTERNAL / spec["folder"]
    return [f for f in spec["required"] if not (root / f).is_file()]


def download(url, dest):
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "har-ml-download-script"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()


def extract_all(zip_path, target):
    """Extract a zip into target, then any zip files found inside it (UCI ships zip-in-zip)."""
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target)
    for _ in range(3):
        inner = [p for p in target.rglob("*.zip") if "__MACOSX" not in p.parts]
        if not inner:
            break
        for p in inner:
            with zipfile.ZipFile(p) as z:
                z.extractall(p.parent)
            p.unlink()
    for junk in target.rglob("__MACOSX"):
        shutil.rmtree(junk, ignore_errors=True)


def normalise_layout(spec):
    """Move the dataset root up if the archive added an extra wrapper directory."""
    root = EXTERNAL / spec["folder"]
    first = spec["required"][0]
    hits = [p for p in root.rglob(Path(first).name) if p.as_posix().endswith(first)]
    if not hits:
        return
    found_root = Path(hits[0].as_posix()[: -len(first)].rstrip("/"))
    if found_root == root:
        return
    for item in found_root.iterdir():
        dest = root / item.name
        if not dest.exists():
            shutil.move(str(item), str(dest))
    while found_root != root and found_root.is_dir() and not any(found_root.iterdir()):
        found_root.rmdir()          # remove the now-empty wrapper directory
        found_root = found_root.parent


def install(name, local_zip=None):
    spec = DATASETS[name]
    print(f"[{name}] {spec['title']}")
    if not missing_files(spec):
        print(f"  already in place: data/external/{spec['folder']}/ (skipped)")
        return True
    target = EXTERNAL / spec["folder"]
    target.mkdir(parents=True, exist_ok=True)
    zip_path = Path(local_zip) if local_zip else EXTERNAL / f"{spec['folder']}.zip"
    try:
        if not local_zip:
            download(spec["url"], zip_path)
        print(f"  extracting into data/external/{spec['folder']}/")
        extract_all(zip_path, target)
        normalise_layout(spec)
    except Exception as exc:  # network errors, bad zip, ...
        print(f"  FAILED: {exc}")
        print(f"  Download it manually from {spec['url']} and run:\n"
              f"    python scripts/download_data.py --from-zip {name} <path to the zip>")
        if target.is_dir() and not any(target.iterdir()):
            target.rmdir()
        return False
    finally:
        if not local_zip and zip_path.exists():
            zip_path.unlink()
    missing = missing_files(spec)
    if missing:
        print(f"  WARNING: files still missing after extraction: {missing}")
        return False
    print("  ok")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=DATASETS, help="download just one dataset")
    ap.add_argument("--from-zip", nargs=2, metavar=("DATASET", "ZIP"), help="install from a local zip instead")
    args = ap.parse_args()
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    if args.from_zip:
        name, path = args.from_zip
        if name not in DATASETS:
            ap.error(f"DATASET must be one of {list(DATASETS)}")
        ok = install(name, path)
    else:
        ok = all([install(n) for n in ([args.only] if args.only else DATASETS)])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
