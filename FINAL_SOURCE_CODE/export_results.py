"""Copy measured evidence into a Git-trackable results directory, excluding model weights."""
import hashlib
import json
import shutil
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    exported = {}
    for run in sorted((root / "runs").iterdir()):
        if not run.is_dir():
            continue
        manifest = run / "manifest.json"
        if not manifest.exists():
            continue
        state = json.loads(manifest.read_text(encoding="utf-8"))
        if state.get("status") != "complete":
            raise RuntimeError(f"Run unfinished: {run.name}")
        destination = root / "results" / run.name
        destination.mkdir(parents=True, exist_ok=True)
        copied = []
        for file in sorted(run.iterdir()):
            if file.suffix in {".json", ".csv", ".png"}:
                shutil.copy2(file, destination / file.name)
                copied.append(file.name)
        checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(run.glob("*.pt"))}
        (destination / "checkpoint_sha256.json").write_text(
            json.dumps(checksums, indent=2), encoding="utf-8")
        exported[run.name] = copied
    (root / "results" / "index.json").write_text(
        json.dumps(exported, indent=2), encoding="utf-8")
    print("Exported:", ", ".join(exported))


if __name__ == "__main__":
    main()
