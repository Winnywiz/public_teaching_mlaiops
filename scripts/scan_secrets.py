"""Scan the current worktree and all reachable Git history for likely credentials.

The scan refuses shallow clones because a clean HEAD says nothing about a secret that
was committed and deleted earlier. GitHub Actions must use checkout with fetch-depth: 0.
Finding output contains only a path, blob identifier, rule, and line number, never a value.
"""
from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path


EXACT_PATTERNS = (
    ("private-key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("aws-access-key", re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("github-token", re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{40,})\b")),
    ("slack-token", re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("google-api-key", re.compile(rb"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("slack-webhook", re.compile(rb"https://hooks\.slack\.com/services/[A-Z0-9/]{20,}")),
    ("azure-storage-key", re.compile(rb"\bAccountKey=([A-Za-z0-9+/=]{40,})")),
)
ASSIGNMENT = re.compile(
    rb"""(?i)\b(?:client[_-]?secret|api[_-]?key|access[_-]?token|refresh[_-]?token|password)\b"""
    rb"""\s*[:=]\s*["']?([A-Za-z0-9_./+=:-]{20,})"""
)
PLACEHOLDER_MARKERS = (
    "replace", "your-", "your_", "example", "dummy", "placeholder", "changeme",
    "change-me", "xxxxx", "<", "\x24{", "$env", "sample", "not-a-real",
)


def git(*args: str, binary: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout if binary else result.stdout.decode("utf-8", errors="replace")


def entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = {char: value.count(char) for char in set(value)}
    return -sum((count / len(value)) * math.log2(count / len(value)) for count in counts.values())


def is_placeholder(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def scan_content(content: bytes) -> list[tuple[int, str]]:
    if b"\0" in content[:8192]:
        return []
    findings: list[tuple[int, str]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        for rule, pattern in EXACT_PATTERNS:
            if pattern.search(line):
                findings.append((line_number, rule))
        match = ASSIGNMENT.search(line)
        if match:
            value = match.group(1).decode("utf-8", errors="ignore").strip().strip("'\"")
            if len(value) >= 24 and entropy(value) >= 3.4 and not is_placeholder(value):
                findings.append((line_number, "credential-assignment"))
    return findings


def history_objects() -> list[tuple[str, str]]:
    output = git("rev-list", "--objects", "--all")
    objects: dict[str, str] = {}
    for line in str(output).splitlines():
        object_id, _, path = line.partition(" ")
        if re.fullmatch(r"[0-9a-f]{40,64}", object_id):
            objects.setdefault(object_id, path or "(unnamed object)")
    return list(objects.items())


def scan_history(objects: list[tuple[str, str]]) -> tuple[int, set[tuple[str, str, int, str]]]:
    findings: set[tuple[str, str, int, str]] = set()
    process = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        for object_id, path in objects:
            process.stdin.write(f"{object_id}\n".encode("ascii"))
            process.stdin.flush()
            header = process.stdout.readline().split()
            if len(header) != 3:
                raise RuntimeError(f"Unexpected git object response for {object_id}")
            object_type = header[1]
            size = int(header[2])
            content = process.stdout.read(size)
            process.stdout.read(1)
            if object_type == b"blob":
                for line, rule in scan_content(content):
                    findings.add((object_id, path, line, rule))
    finally:
        process.stdin.close()
    error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code:
        raise RuntimeError(error.strip() or "git cat-file failed")
    return len(objects), findings


def scan_worktree(root: Path) -> set[tuple[str, str, int, str]]:
    raw_paths = git("ls-files", "-co", "--exclude-standard", "-z", binary=True)
    findings: set[tuple[str, str, int, str]] = set()
    for raw_path in bytes(raw_paths).split(b"\0"):
        if not raw_path:
            continue
        relative = Path(raw_path.decode("utf-8", errors="replace"))
        path = root / relative
        if not path.is_file():
            continue
        try:
            content = path.read_bytes()
        except OSError:
            continue
        for line, rule in scan_content(content):
            findings.add(("working-tree", relative.as_posix(), line, rule))
    return findings


def main() -> int:
    try:
        root = Path(str(git("rev-parse", "--show-toplevel")).strip()).resolve()
        shallow = str(git("rev-parse", "--is-shallow-repository")).strip().lower() == "true"
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"secret scan could not inspect Git: {exc}", file=sys.stderr)
        return 2

    if shallow:
        print(
            "secret scan refused a shallow clone; use actions/checkout with fetch-depth: 0 "
            "or run git fetch --unshallow",
            file=sys.stderr,
        )
        return 2

    try:
        blob_count, history_findings = scan_history(history_objects())
        findings = history_findings | scan_worktree(root)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"secret scan could not inspect repository history: {exc}", file=sys.stderr)
        return 2

    if findings:
        print("Potential credentials found. Values were intentionally omitted:")
        for source, path, line, rule in sorted(findings):
            location = f"blob {source[:12]}" if source != "working-tree" else "working tree"
            print(f"  {path}:{line} [{rule}; {location}]")
        print("Rotate any exposed credential. Deleting it from the latest commit is not enough.")
        return 1

    print(
        f"Secret scan passed: {blob_count} reachable Git objects and the current worktree "
        f"checked from {root}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
