import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count

from Bio.Align import PairwiseAligner

sys.path.append("/data/atran16/ProteinClassification_3D/evaluations")
from SequentialSimilarity import fetch_fasta


THRESHOLD     = 20.0
PROTEINS_DIR  = "/data/atran16/ProteinClassification_3D/3D_PDB_5013/PNG126"
CACHE_PATH    = "/data/atran16/ProteinClassification_3D/visuallization/fasta_cache.json"
RESULT_PATH   = "/data/atran16/ProteinClassification_3D/3D_PDB_5013/protein_neighbors_5013_20percent.json"
SAVE_EVERY    = 100   # save partial result every N pairs evaluated


# ---------------- FASTA caching ----------------
def load_or_fetch_fastas(pdb_ids, cache_path: str, n_threads: int = 16) -> dict:
    """Return {pdb_id: sequence}. Fetches missing entries, persists to disk."""
    cache: dict = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    missing = [p for p in pdb_ids if p not in cache]
    print(f"FASTA cache: {len(cache)} cached / {len(missing)} to fetch")

    if missing:
        def _fetch(p):
            try:
                return p, fetch_fasta(p)
            except Exception as e:
                print(f"  [warn] {p}: {e}")
                return p, None

        done = 0
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            for p, seq in ex.map(_fetch, missing):
                cache[p] = seq
                done += 1
                if done % 100 == 0:
                    print(f"  fetched {done}/{len(missing)}")
                    with open(cache_path, "w") as f:
                        json.dump(cache, f)

        with open(cache_path, "w") as f:
            json.dump(cache, f)

    return cache


# ---------------- Alignment (CPU work) ----------------
def _percent_identity(seq1: str, seq2: str) -> float:
    aligner = PairwiseAligner()
    aligner.mode             = "global"
    aligner.match_score      = 1
    aligner.mismatch_score   = 0
    aligner.open_gap_score   = -1
    aligner.extend_gap_score = -0.5
    best   = next(iter(aligner.align(seq1, seq2)))
    counts = best.counts()
    align_len = counts.identities + counts.mismatches + counts.gaps
    return (counts.identities / align_len) * 100 if align_len > 0 else 0.0


def _pair_worker(args):
    p1, p2, seq1, seq2 = args
    try:
        sim = _percent_identity(seq1, seq2)
    except Exception:
        return None
    if sim >= THRESHOLD:
        return (p1, p2, round(sim, 2))
    return None


def _gen_pairs(valid_ids, fastas, done_pairs):
    """Yield (p1, p2, seq1, seq2) for every unseen unordered pair."""
    n = len(valid_ids)
    for i in range(n):
        p1 = valid_ids[i]
        seq1 = fastas[p1]
        for j in range(i + 1, n):
            p2 = valid_ids[j]
            if (p1, p2) in done_pairs:
                continue
            yield (p1, p2, seq1, fastas[p2])


# ---------------- Main ----------------
def main():
    proteins = sorted(
        p for p in os.listdir(PROTEINS_DIR)
        if os.path.isdir(os.path.join(PROTEINS_DIR, p))
    )
    print(f"Total proteins: {len(proteins)}")

    fastas = load_or_fetch_fastas(proteins, CACHE_PATH)
    valid_ids = sorted(p for p in proteins if fastas.get(p))
    print(f"Proteins with valid sequences: {len(valid_ids)}\n")

    # Resume: load existing neighbor map (and reconstruct done_pairs from it)
    neighbors: dict = {}
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH) as f:
            neighbors = json.load(f)
        print(f"Loaded existing result with {len(neighbors)} entries")

    done_pairs = set()
    for p1, nbrs in neighbors.items():
        for p2 in nbrs:
            key = (p1, p2) if p1 < p2 else (p2, p1)
            done_pairs.add(key)

    n           = len(valid_ids)
    total_pairs = n * (n - 1) // 2
    print(f"Total pairs to evaluate: {total_pairs:,}")

    pair_count = 0
    hit_count  = 0
    workers    = max(1, cpu_count() - 1)

    with Pool(workers) as pool:
        for result in pool.imap_unordered(
                _pair_worker,
                _gen_pairs(valid_ids, fastas, done_pairs),
                chunksize=200):

            pair_count += 1

            if result is not None:
                p1, p2, sim = result
                neighbors.setdefault(p1, {})[p2] = sim
                neighbors.setdefault(p2, {})[p1] = sim
                hit_count += 1

            if pair_count % SAVE_EVERY == 0:
                with open(RESULT_PATH, "w") as f:
                    json.dump(neighbors, f, indent=2)
                print(f"  [{pair_count:>10,}/{total_pairs:,}] "
                      f"hits={hit_count:,}  proteins_with_neighbors={sum(1 for v in neighbors.values() if v):,}")

    # Make sure every protein has an entry (even if empty)
    for p in valid_ids:
        neighbors.setdefault(p, {})

    with open(RESULT_PATH, "w") as f:
        json.dump(neighbors, f, indent=2)

    print(f"\nDone. Pairs evaluated: {pair_count:,}  Hits: {hit_count:,}")
    print(f"Saved neighbor map -> {RESULT_PATH}")


if __name__ == "__main__":
    main()