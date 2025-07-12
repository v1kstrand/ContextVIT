from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def delete_in_parallel(target, num_threads=4):
    """Dummy helper to remove files under target path"""
    path = Path(target)
    if not path.exists():
        return

    def remove(p):
        try:
            p.unlink()
        except Exception:
            pass

    files = [p for p in path.iterdir() if p.is_file()]
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        for f in files:
            ex.submit(remove, f)
