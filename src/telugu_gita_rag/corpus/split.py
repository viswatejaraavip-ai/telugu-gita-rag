"""Split verses (not queries) into train / held-out.

Holding out VERSES is what makes the eval honest: held-out queries generated
from training verses would measure memorisation, not generalisation.
"""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval import load_corpus

OUT = Path(__file__).resolve().parent.parent / "finetune"
HELD_FRAC = 0.15
SEED = 13

def main():
    recs = load_corpus()
    # Keep the 14 hand-written eval questions' gold verses OUT of training too.
    evalf = OUT.parent / "eval" / "questions.jsonl"
    protected = set()
    if evalf.exists():
        for l in evalf.open(encoding="utf-8"):
            if l.strip():
                protected.update(json.loads(l)["gold_ids"])

    pool = [r for r in recs if r["id"] not in protected]
    random.Random(SEED).shuffle(pool)
    n_held = int(len(recs) * HELD_FRAC)
    held = pool[:n_held]
    train = pool[n_held:]

    OUT.mkdir(exist_ok=True)
    for name, rows in (("train_verses", train), ("held_verses", held)):
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"corpus            : {len(recs)}")
    print(f"protected (eval gold, excluded from both): {len(protected)}")
    print(f"train verses      : {len(train)}")
    print(f"held-out verses   : {len(held)}")
    overlap = {r['id'] for r in train} & {r['id'] for r in held}
    print(f"train/held overlap: {len(overlap)}  (must be 0)")
    print(f"protected in train: {len({r['id'] for r in train} & protected)}  (must be 0)")

if __name__ == "__main__":
    main()
