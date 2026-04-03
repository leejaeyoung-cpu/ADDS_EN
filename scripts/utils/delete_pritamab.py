"""
delete_pritamab.py  —  Remove all Pritamab files from GitHub repo
"""
import os
import time
from github import Github, Auth, GithubException

TOKEN = os.environ.get("GITHUB_TOKEN", "")  # export GITHUB_TOKEN=ghp_...
if not TOKEN:
    raise EnvironmentError("GITHUB_TOKEN environment variable not set")
REPO  = "leejaeyoung-cpu/ADDS"
auth  = Auth.Token(TOKEN)
g     = Github(auth=auth)
repo  = g.get_repo(REPO)

def delete_path(remote_path: str):
    """Delete a single file on GitHub."""
    try:
        f = repo.get_contents(remote_path, ref="main")
        repo.delete_file(remote_path, "remove: delete Pritamab files (out of scope)", f.sha, branch="main")
        print(f"  DELETE  {remote_path}")
        time.sleep(0.5)
    except GithubException as e:
        if e.status == 404:
            print(f"  SKIP(404) {remote_path}")
        else:
            print(f"  ERROR  {remote_path}: {e}")

def delete_folder(folder_path: str):
    """Recursively delete all files in a GitHub folder."""
    try:
        items = repo.get_contents(folder_path, ref="main")
        if not isinstance(items, list):
            items = [items]
        for item in items:
            if item.type == "dir":
                delete_folder(item.path)
            else:
                delete_path(item.path)
    except GithubException as e:
        print(f"  ERROR listing {folder_path}: {e}")

print("=== Removing Pritamab files from GitHub ===\n")

# 1. analysis/pritamab/ 전체 삭제
print("[1] analysis/pritamab/")
delete_folder("analysis/pritamab")

# 2. figures/pritamab/ 전체 삭제
print("\n[2] figures/pritamab/")
delete_folder("figures/pritamab")

# 3. figures/ 루트의 pritamab_*.png 삭제
print("\n[3] figures/pritamab_*.png")
try:
    root_figs = repo.get_contents("figures", ref="main")
    pritamab_figs = [f for f in root_figs
                     if f.type == "file" and "pritamab" in f.name.lower()]
    for f in pritamab_figs:
        delete_path(f.path)
    kras_figs = [f for f in root_figs
                 if f.type == "file" and "kras" in f.name.lower()]
    for f in kras_figs:
        delete_path(f.path)
except GithubException as e:
    print(f"  ERROR: {e}")

print("\n=== Done ===")
