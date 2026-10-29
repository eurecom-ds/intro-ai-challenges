#!/usr/bin/env python3
"""Prepare versioned public helpers and separate private Snake evaluation seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import secrets
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import protocol


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def entry(path, relative):
    data = path.read_bytes()
    return {"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def prepare_public():
    files = {name: entry(ROOT / name, name)
             for name in ("environment.py", "evaluator.py", "protocol.py", "model.py")}
    manifest = {"challenge_version": "snake-v1", "protocol_fingerprint": protocol.protocol_fingerprint(),
                "files": files, "public_seed_pairs": protocol.public_seed_pairs(),
                "release_status": "pending_instructor_colab_rehearsal"}
    path = ROOT / "data" / "manifest.json"
    write_json(path, manifest)
    print("Public manifest:", path)
    print("SHA-256:", hashlib.sha256(path.read_bytes()).hexdigest())


def prepare_private(directory):
    directory = Path(directory).resolve()
    if directory == ROOT or ROOT in directory.parents:
        raise ValueError("Choose a private output directory outside the public Snake activity.")
    seed_path = directory / "private_seeds.json"
    manifest_path = directory / "servermanifest.json"
    if seed_path.exists() or manifest_path.exists():
        raise ValueError("Evaluation assets already exist. Keep them fixed; use a new directory for a new protocol.")
    occupied = {int(seed) for pair in protocol.public_seed_pairs() for seed in pair}
    demo = getattr(protocol, "DEMO_SEED_PAIR", ())
    occupied.update(int(seed) for seed in demo)
    pairs = []
    for _ in range(100):
        pair = []
        for _ in range(2):
            while True:
                seed = secrets.randbits(32)
                if seed not in occupied:
                    occupied.add(seed)
                    pair.append(seed)
                    break
        pairs.append(pair)
    write_json(seed_path, pairs)
    seed_path.chmod(0o600)
    manifest = {"challenge_version": "snake-v1", "protocol_fingerprint": protocol.protocol_fingerprint(),
                "evaluation_episodes": 100, "horizon": 300,
                "files": {"seeds": entry(seed_path, "private_seeds.json")}}
    write_json(manifest_path, manifest)
    manifest_path.chmod(0o600)
    print("Private evaluation assets prepared:", directory)


def check_copies(leaderboard_root):
    destination = Path(leaderboard_root) / "challenges" / "snake"
    for name in ("environment.py", "evaluator.py", "protocol.py"):
        if (ROOT / name).read_bytes() != (destination / name).read_bytes():
            raise ValueError("Notebook/server helper differs: " + name)
    print("Simulator, evaluator and protocol copies match.")


def prepare_bundle(output):
    output = Path(output).resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Keep rehearsal bundles outside the public activity directory.")
    manifest_path = ROOT / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["protocol_fingerprint"] != protocol.protocol_fingerprint():
        raise ValueError("Refresh the public manifest before making a helper bundle.")
    names = ("environment.py", "evaluator.py", "protocol.py", "model.py")
    for name in names:
        if entry(ROOT / name, name) != manifest["files"][name]:
            raise ValueError("The public helper differs from its manifest: " + name)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in (*names, "data/manifest.json"):
            archive.write(ROOT / name, "snake-helpers/" + name)
    print("Verified rehearsal helper bundle:", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("public", help="Refresh public helper hashes after helper changes.")
    private = commands.add_parser("private", help="Create a new private evaluation seed set.")
    private.add_argument("--private-dir", type=Path, required=True)
    check = commands.add_parser("check", help="Check that notebook and server helper copies match.")
    check.add_argument("--leaderboard-root", type=Path, required=True)
    bundle = commands.add_parser("bundle", help="Package verified helpers for an unpublished Colab rehearsal.")
    bundle.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "public":
        prepare_public()
    elif args.command == "private":
        prepare_private(args.private_dir)
    elif args.command == "check":
        check_copies(args.leaderboard_root)
    else:
        prepare_bundle(args.output)


if __name__ == "__main__":
    main()
