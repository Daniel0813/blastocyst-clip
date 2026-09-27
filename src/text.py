"""Text side: Gardner class prompts and label-derived pseudo-captions.

    from src.text import load_bank, class_prompts, pseudo_caption

class_prompts(bank, "EXP") -> {code: [prompt, ...]}  (every phrasing x template,
    for prompt ensembling: encode all, average per class)
pseudo_caption(bank, exp, icm, te, rng) -> one sentence describing a grade triple,
    used to build image-text pairs from the grade labels for fine-tuning.
"""
import json
import random
from pathlib import Path

ND = 3
BANK_PATH = Path(__file__).resolve().parent.parent / "prompts" / "gardner_prompts.json"


def load_bank(path=BANK_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def class_prompts(bank: dict, grade: str) -> dict[int, list[str]]:
    """All prompts per class code for one grade ('EXP', 'ICM' or 'TE')."""
    out = {}
    for code, phrases in bank[grade].items():
        if grade != "EXP":
            phrases = [f"a blastocyst in which {p}" for p in phrases]
        out[int(code)] = [t.format(p) for p in phrases for t in bank["templates"]]
    return out


def pseudo_caption(bank: dict, exp: int, icm: int, te: int, rng: random.Random) -> str:
    """Grade triple -> caption, e.g. (3, 1, 0) -> 'This image shows an expanded ...'."""
    parts = [f"This image shows {rng.choice(bank['EXP'][str(exp)])}."]
    if icm == ND or te == ND:
        nd = rng.choice(bank["ND"])
        parts.append(nd[0].upper() + nd[1:] + ".")
    else:
        icm_c = rng.choice(bank["ICM"][str(icm)])
        te_c = rng.choice(bank["TE"][str(te)])
        parts.append(f"{icm_c[0].upper() + icm_c[1:]}, and {te_c}.")
    return " ".join(parts)


if __name__ == "__main__":
    bank = load_bank()
    for g in ("EXP", "ICM", "TE"):
        p = class_prompts(bank, g)
        print(g, {k: len(v) for k, v in p.items()}, "| e.g.", p[0][1])
    rng = random.Random(0)
    for triple in [(0, 3, 3), (2, 0, 0), (3, 1, 0), (4, 2, 1)]:
        print(triple, "->", pseudo_caption(bank, *triple, rng))
