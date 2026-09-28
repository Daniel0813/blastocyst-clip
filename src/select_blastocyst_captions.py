"""Select the blastocyst-stage captions from the expert caption dataset.

The caption images are Gomez et al. time-lapse frames (EmbryoScope), named
<video>_WELL<n>_RUN<frame>.jpeg. A frame's developmental stage comes from the
Gomez stage annotations; tSB/tB/tEB/tHB are blastocyst stages. The annotations
are downloaded to a temporary folder and discarded (they are not project data).

Writes data/captions/blastocyst_captions.jsonl with, per caption: source set,
stage, and any Gardner grades stated in the text (regex extraction, noisy —
e.g. one caption says "expanded blastocyst with the Gardner grade 3AB").

Usage:
    python src/select_blastocyst_captions.py [--captions-dir data/captions]
"""
import argparse
import csv
import io
import json
import random
import re
import tarfile
import urllib.request
from pathlib import Path

HOLDOUT_FRACTION = 0.30
HOLDOUT_SEED = 42
ANNOTATIONS_URL ="https://zenodo.org/records/6390798/files/embryo_dataset_annotations.tar.gz?download=1"
BLASTOCYST_STAGES = {"tSB", "tB", "tEB", "tHB"}
NAME = re.compile(r"^(.+)_WELL\d+_RUN(\d+)\.jpe?g$", re.I)

# Expansion grade: "Gardner grade 3", "(Gardner grade 4)", "grade 3AB"
EXP_RE = re.compile(r"gardner[^.]{0,20}?\b([1-6])(?:[ABC]{2})?\b|\bgrade\s*([1-6])(?:[ABC]{2})?\b", re.I)
# Combined code like 4AB
COMBINED_RE = re.compile(r"\b([1-6])([ABC])([ABC])\b")
ICM_RE = re.compile(r"(?:inner cell mass|\bICM\b)[^.]{0,60}?\b(?:grade|quality|type)\s*([ABC])\b", re.I)
TE_RE = re.compile(r"(?:trophectoderm|\bTE\b)[^.]{0,60}?\b(?:grade|quality|type)\s*([ABC])\b", re.I)


def load_stages() -> dict:
    """{video: [(phase, first_frame, last_frame), ...]} from the Gomez annotations."""
    with urllib.request.urlopen(ANNOTATIONS_URL) as r:
        data = r.read()
    stages = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar.getmembers():
            if not m.name.endswith("_phases.csv"):
                continue
            video = Path(m.name).name[: -len("_phases.csv")]
            text = tar.extractfile(m).read().decode()
            stages[video] = [(r[0].strip(), int(r[1]), int(r[2])) for r in csv.reader(io.StringIO(text)) if r]
    return stages


def stage_of(filename: str, stages: dict):
    m = NAME.match(filename)
    if not m or m.group(1) not in stages:
        return None
    frame = int(m.group(2))
    return next((p for p, a, b in stages[m.group(1)] if a <= frame <= b), None)


def extract_grades(caption: str) -> dict:
    out = {"exp_grade": None, "icm_grade": None, "te_grade": None}
    if m := COMBINED_RE.search(caption):
        out.update(exp_grade=int(m.group(1)), icm_grade=m.group(2), te_grade=m.group(3))
    elif m := EXP_RE.search(caption):
        out["exp_grade"] = int(m.group(1) or m.group(2))
    if out["icm_grade"] is None and (m := ICM_RE.search(caption)):
        out["icm_grade"] = m.group(1).upper()
    if out["te_grade"] is None and (m := TE_RE.search(caption)):
        out["te_grade"] = m.group(1).upper()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions-dir", default="data/captions", type=Path)
    args = ap.parse_args()

    stages = load_stages()
    selected = []
    for source in ("silver", "gold"):
        path = args.captions_dir / f"caption_{source}.jsonl"
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            stage = stage_of(row["filename"], stages)
            if stage in BLASTOCYST_STAGES:
                selected.append({"filename": row["filename"], "source": source, "stage": stage,
                                 **extract_grades(row["caption"]), "caption": row["caption"].strip()})

    # Held-out split for cross-clinic testing, fixed before any training: whole
    # embryos (videos) are held out until ~30% of captions are covered.
    videos = sorted({NAME.match(r["filename"]).group(1) for r in selected})
    rng = random.Random(HOLDOUT_SEED)
    rng.shuffle(videos)
    held, count = set(), 0
    for v in videos:
        if count >= HOLDOUT_FRACTION * len(selected):
            break
        held.add(v)
        count += sum(NAME.match(r["filename"]).group(1) == v for r in selected)
    for r in selected:
        r["holdout"] = NAME.match(r["filename"]).group(1) in held

    out = args.captions_dir / "blastocyst_captions.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(selected)
    print(f"blastocyst-stage captions: {n} "
          f"(silver {sum(r['source'] == 'silver' for r in selected)}, gold {sum(r['source'] == 'gold' for r in selected)})")
    for key in ("exp_grade", "icm_grade", "te_grade"):
        vals = [r[key] for r in selected if r[key] is not None]
        dist = {v: vals.count(v) for v in sorted(set(vals), key=str)}
        print(f"  {key}: stated in {len(vals)} -> {dist}")
    hold = [r for r in selected if r["holdout"]]
    print(f"held out for cross-clinic testing: {len(hold)} captions from "
          f"{len({NAME.match(r['filename']).group(1) for r in hold})} embryos "
          f"({sum(r['exp_grade'] is not None for r in hold)} state an expansion grade)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
