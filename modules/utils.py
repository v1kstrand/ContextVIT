import os
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys


def install_if_missing(package: str):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

class PyDrive():
    def __init__(self):
        GoogleAuth, GoogleDrive = self.init()
        self.gauth = GoogleAuth()
        self.gauth.CommandLineAuth()   
        self.drive = GoogleDrive(self.gauth)
        
    def init(self):
        install_if_missing("PyDrive2")
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive   
        return GoogleAuth, GoogleDrive   

    def move_to_gdrive(self, save_name, file_path):
        gfile = self.drive.CreateFile({'title': save_name})
        gfile.SetContentFile(file_path, file_path)
        gfile.Upload()
        print(f"Successfully uploaded {file_path} and saved as {save_name}")
        
    def download_from_gdrive(self, file_id: str, dest_path: str):
        """
        Download a file from Google Drive to the local filesystem.

        Args:
            file_id (str): The ID of the file on Drive (e.g., '1AbCdEfGhIjKlMnOp').
            dest_path (str): Local path where to save the downloaded file.
        """
        gfile = self.drive.CreateFile({'id': file_id})
        gfile.GetContentFile(dest_path)
        print(f"Successfully downloaded file to {dest_path}")

def parallel_collect_paths(root_path: str, num_threads: int = 8):
    """
    Walk the directory tree under `root_path` in parallel, collecting:
      - all_files: a list of full paths to every file
      - all_dirs:  a list of full paths to every directory (excluding root_path)
    Finally, the caller can append root_path to all_dirs if needed.

    Args:
        root_path (str): The top directory to traverse.
        num_threads (int): How many worker threads to spawn.

    Returns:
        all_files (List[str]) : full paths of all files under root_path
        all_dirs  (List[str]) : full paths of all subdirectories under root_path
    """
    # Thread‐safe queue of directories to process
    q = Queue()
    q.put(root_path)

    # Shared lists (protected by `lock`)
    all_files = []
    all_dirs = []
    lock = threading.Lock()

    def worker():
        while True:
            try:
                dirpath = q.get_nowait()
            except Empty:
                # No more directories to process
                return

            try:
                # List all entries in this directory
                entries = os.listdir(dirpath)
            except Exception:
                # If we can’t read this directory for any reason, skip it
                q.task_done()
                continue

            for name in entries:
                full_path = os.path.join(dirpath, name)
                if os.path.isdir(full_path):
                    # Record this directory, then schedule it for further walking
                    with lock:
                        all_dirs.append(full_path)
                    q.put(full_path)
                else:
                    # It’s a file; record it
                    with lock:
                        all_files.append(full_path)

            q.task_done()

    # Spawn worker threads
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)

    # Wait until every directory has been processed
    q.join()

    # (Optional) Join threads (they should all exit once queue is empty)
    for t in threads:
        t.join()

    return all_files, all_dirs

def delete_in_parallel(root_path: str, num_threads: int = 8):
    # 1) Gather everything in parallel
    files_list, dirs_list = parallel_collect_paths(root_path, num_threads=num_threads)

    # 2) Delete all files in parallel
    def _remove_file(path):
        try:
            os.remove(path)
        except Exception as e:
            print(f"Failed to remove file {path!r}: {e}")

    if files_list:
        print(f"Deleting {len(files_list)} files ...")
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            list(pool.map(_remove_file, files_list))

    # 3) Delete all directories in parallel (deepest first)
    # Sort by depth: deeper dirs (more separators) first
    dirs_sorted = sorted(dirs_list, key=lambda p: -p.count(os.sep))

    def _remove_dir(path):
        try:
            os.rmdir(path)
        except Exception as e:
            print(f"Failed to remove directory {path!r}: {e}")

    if dirs_sorted:
        print(f"Deleting {len(dirs_sorted)} directories ...")
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            list(pool.map(_remove_dir, dirs_sorted))