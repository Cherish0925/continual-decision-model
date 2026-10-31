"""Package scientific-task runs without regenerable features or pretrained weights.

Run from the repository root. Original experiment files are never modified.
The archives restore project-relative paths. The default selection root is
runs/discoveryworld/; --base supports other experiment directories.
"""
from __future__ import annotations

import argparse
import filecmp
import gzip
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile


CHECKPOINTS = {"last.pt", "best.pt", "initial.pt", "critic.pt", "policy.pt", "belief.pt"}
REFERENCE = {
    "measurement_memory/action_run/last.pt",
    "calibration/mixture/shared_seed31/last.pt",
    *(f"belief_actions/{name}/last.pt" for name in
      ("full_seed61", "full_seed62", "none_seed61", "none_seed62", "observations_seed61")),
}
TEXT = {".json", ".jsonl", ".csv", ".txt", ".md", ".py", ".yaml", ".yml"}
IMAGES = {".png", ".jpg", ".jpeg"}
SECRET = re.compile(rb"(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)")


def files(root, runs=(), base="runs/discoveryworld"):
    base = root / base
    for selected in ([base / run for run in runs] if runs else [base]):
        if selected.is_file() and not selected.is_symlink():
            yield selected
            continue
        if not selected.is_dir():
            raise FileNotFoundError(selected)
        for directory, names, filenames in os.walk(selected):
            names[:] = sorted(n for n in names if n not in {"features", "__pycache__", "contexts"})
            for name in sorted(filenames):
                path = Path(directory) / name
                if path.is_symlink():
                    continue
                yield path


def portable(value, root):
    if isinstance(value, str):
        # Task worktrees share the same project-relative runs directory.
        value = re.sub(re.escape(str(root)) + r"/\.worktrees/[^/]+/", "", value)
        value = value.replace(str(root) + "/.venv/bin/python", "python")
        return value.replace(str(root) + "/", "").replace(str(root), ".")
    if isinstance(value, dict):
        return {k: portable(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v, root) for v in value]
    if isinstance(value, tuple):
        return tuple(portable(v, root) for v in value)
    return value


def package(root, output, kind, runs=(), name="cdm-scientific", base="runs/discoveryworld"):
    import torch

    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{name}-{kind}.tar.zst"
    for previous_part in output.glob(archive.name + ".part*"):
        previous_part.unlink()
    inventory, repeated_files = [], {}
    with archive.open("wb") as target:
        compressor = subprocess.Popen(["zstd", "-q", "-T4", "-6", "--long=30", "-c"], stdin=subprocess.PIPE, stdout=target)
        with tarfile.open(fileobj=compressor.stdin, mode="w|") as tar:
            for path in files(root, runs, base):
                checkpoint = path.name in CHECKPOINTS or bool(re.fullmatch(r"(?:epoch|update|checkpoint)\d+\.pt", path.name))
                image = path.suffix.lower() in IMAGES
                compressed_text = path.name.endswith((".json.gz", ".jsonl.gz"))
                record = path.suffix in TEXT or compressed_text or path.suffix in {".svg", ".pdf", ".npz"}
                reference = path.relative_to(root).as_posix() in {
                    f"runs/discoveryworld/{item}" for item in REFERENCE}
                if not {"reference": reference, "models": checkpoint, "inference": checkpoint,
                        "observations": image, "records": record}[kind]:
                    continue
                relative = path.relative_to(root).as_posix()
                info = tar.gettarinfo(str(path), arcname=relative)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                # Paired evaluations copy the same simulator randomness into
                # every arm. Preserve those paths without storing the bytes
                # again; tar restores the links as ordinary readable files.
                if image or compressed_text:
                    candidates = repeated_files.setdefault((path.suffix, info.size), [])
                    previous = next((p for p in candidates if filecmp.cmp(p, path, shallow=False)), None)
                    if previous is not None:
                        info.type = tarfile.LNKTYPE
                        info.linkname = previous.relative_to(root).as_posix()
                        info.size = 0
                        tar.addfile(info)
                        inventory.append(relative)
                        continue
                    candidates.append(path)
                if checkpoint:
                    # Change metadata paths only; tensor values and training results stay intact.
                    saved = portable(torch.load(path, map_location="cpu", weights_only=True), root)
                    if kind == "inference":
                        saved.pop("optimizer", None)
                    with tempfile.TemporaryFile() as stream:
                        torch.save(saved, stream)
                        info.size = stream.tell()
                        stream.seek(0)
                        tar.addfile(info, stream)
                elif image:
                    with path.open("rb") as stream:
                        tar.addfile(info, stream)
                else:
                    data = path.read_bytes()
                    compressed = compressed_text
                    original = data
                    content = gzip.decompress(data) if compressed else data
                    if SECRET.search(content):
                        raise ValueError(f"Credential-like content in {relative}; exclude before publication")
                    if path.suffix in TEXT or compressed:
                        data = portable(content.decode("utf-8"), root).encode("utf-8")
                    if compressed:
                        data = original if data == content else gzip.compress(data, compresslevel=6, mtime=0)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
                inventory.append(relative)
        compressor.stdin.close()
        if compressor.wait() != 0:
            raise RuntimeError("Archive compression failed")
    # Separate pieces keep every asset below the GitHub per-file limit.
    limit = 1800 * 1024**2
    if archive.stat().st_size > limit:
        with archive.open("rb") as source:
            part = 0
            while source.peek(1):
                piece = archive.with_name(archive.name + f".part{part:02d}")
                with piece.open("wb") as destination:
                    remaining = limit
                    while remaining:
                        block = source.read(min(8 * 1024**2, remaining))
                        if not block:
                            break
                        destination.write(block)
                        remaining -= len(block)
                part += 1
        archive.unlink()
    listing = f"{kind}-files.txt" if name == "cdm-scientific" else f"{name}-{kind}-files.txt"
    (output / listing).write_text("\n".join(inventory) + "\n")
    print(json.dumps({"kind": kind, "files": len(inventory), "assets": [
        {"name": p.name, "bytes": p.stat().st_size}
        for p in sorted(output.glob(f"{name}-{kind}.tar.zst*"))]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("reference", "models", "inference", "records", "observations"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("runs/discoveryworld"), help="Project-relative selection root")
    parser.add_argument("--run", action="append", default=[], help="Run directory or file relative to --base; repeat for selected release contents")
    parser.add_argument("--name", default="cdm-scientific", help="Archive filename prefix")
    args = parser.parse_args()
    package(Path.cwd().resolve(), args.output, args.kind, args.run, args.name, args.base)
