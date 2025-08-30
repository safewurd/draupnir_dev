# dir_map.py
import argparse
import fnmatch
import os
import sys
from datetime import datetime
from typing import Iterable, List, Tuple

DEFAULT_EXCLUDES = [
    ".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".ipynb_checkpoints", ".DS_Store", "node_modules",
    ".idea", ".vscode", "dist", "build", ".streamlit/__pycache__",
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print a directory tree (project map)."
    )
    p.add_argument("root", nargs="?", default=".", help="Root directory (default: .)")
    p.add_argument("--include-files", action="store_true", help="Include files (not just folders).")
    p.add_argument("--max-depth", type=int, default=0, help="Limit depth (0 = unlimited).")
    p.add_argument("--exclude", default=",".join(DEFAULT_EXCLUDES),
                   help="Comma-separated patterns/paths to exclude.")
    p.add_argument("--out", default="", help="Write output to this file (txt/md).")
    p.add_argument("--show-counts", action="store_true", help="Append counts summary.")
    return p.parse_args()

def should_exclude(path: str, patterns: List[str]) -> bool:
    base = os.path.basename(path)
    if base.startswith(".") and base not in (".", ".."):
        # hidden files/dirs often unwanted; rely on patterns to keep if needed
        pass
    for pat in patterns:
        pat = pat.strip()
        if not pat:
            continue
        # match either by basename or by full relative path
        if fnmatch.fnmatch(base, pat) or fnmatch.fnmatch(path.replace("\\", "/"), pat):
            return True
    return False

def list_entries(dir_path: str, exclude_patterns: List[str]) -> Tuple[List[os.DirEntry], List[os.DirEntry]]:
    try:
        entries = list(os.scandir(dir_path))
    except PermissionError:
        return [], []
    dirs = [e for e in entries if e.is_dir(follow_symlinks=False) and not should_exclude(
        os.path.relpath(e.path), exclude_patterns)]
    files = [e for e in entries if e.is_file(follow_symlinks=False) and not should_exclude(
        os.path.relpath(e.path), exclude_patterns)]
    dirs.sort(key=lambda e: e.name.lower())
    files.sort(key=lambda e: e.name.lower())
    return dirs, files

def build_tree(root: str, include_files: bool, max_depth: int, exclude_patterns: List[str]) -> Tuple[List[str], int, int]:
    lines: List[str] = []
    dir_count = 0
    file_count = 0
    root = os.path.abspath(root)
    header = f"{os.path.basename(root) or root}"
    lines.append(header)

    def recurse(current: str, prefix: str, depth: int):
        nonlocal dir_count, file_count
        if max_depth and depth >= max_depth:
            # indicate there is more below
            dirs, files = list_entries(current, exclude_patterns)
            more = len(dirs) + (len(files) if include_files else 0)
            if more:
                lines.append(prefix + "└── …")
            return

        dirs, files = list_entries(current, exclude_patterns)
        children: List[Tuple[os.DirEntry, bool]] = [(d, True) for d in dirs]
        if include_files:
            children += [(f, False) for f in files]

        for idx, (entry, is_dir) in enumerate(children):
            connector = "└── " if idx == len(children) - 1 else "├── "
            line = prefix + connector + entry.name + ("/" if is_dir else "")
            lines.append(line)

            if is_dir:
                dir_count += 1
                extension = "    " if idx == len(children) - 1 else "│   "
                recurse(entry.path, prefix + extension, depth + 1)
            else:
                file_count += 1

    recurse(root, "", 0)
    return lines, dir_count, file_count

def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    exclude_patterns = [p.strip() for p in args.exclude.split(",")] if args.exclude else []

    lines, dcount, fcount = build_tree(
        root=root,
        include_files=args.include_files,
        max_depth=max(0, args.max_depth),
        exclude_patterns=exclude_patterns,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = [f"# Project Map: {root}", f"Generated: {timestamp}", ""]
    output = header + (["```"] if args.out.endswith(".md") else []) + lines + (["```"] if args.out.endswith(".md") else [])
    if args.show_counts:
        output += ["", f"Summary: {dcount} folder(s), {fcount} file(s)"]

    text = "\n".join(output)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
        print(f"Wrote: {args.out}")
    else:
        print(text)

if __name__ == "__main__":
    sys.exit(main())
