import requests
from Bio.Align import PairwiseAligner


def fetch_fasta(pdb_id: str) -> str:
    """Fetch the FASTA sequence for a PDB ID from RCSB."""
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}/download"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise ValueError(f"Could not fetch FASTA for PDB ID '{pdb_id}' (HTTP {response.status_code})")

    lines = response.text.strip().split("\n")
    sequence = "".join(line.strip() for line in lines if not line.startswith(">"))
    if not sequence:
        raise ValueError(f"Empty sequence returned for PDB ID '{pdb_id}'")
    return sequence


def similarity_score(pdb_id1: str, pdb_id2: str, verbose = 1 ) -> float:
    """
    Compute the % sequence identity between two proteins given their PDB IDs.

    Uses global pairwise alignment (Needleman-Wunsch).
    % Identity = identical positions / alignment length * 100

    >30% identity → proteins likely share almost identical 3D structures (same family).
    <30% identity → proteins likely have divergent structures.

    Args:
        pdb_id1: First PDB ID  (e.g. "1ABC")
        pdb_id2: Second PDB ID (e.g. "2XYZ")

    Returns:
        Percent identity as a float (0.0 – 100.0)
    """
    if verbose:
        print(f"Fetching sequence for {pdb_id1.upper()} ...")
    seq1 = fetch_fasta(pdb_id1)
    if verbose:
        print(f"Fetching sequence for {pdb_id2.upper()} ...")
    seq2 = fetch_fasta(pdb_id2)

    if verbose:
        print(f"\n{pdb_id1.upper()} sequence length: {len(seq1)}")
        print(f"{pdb_id2.upper()} sequence length: {len(seq2)}")
        print("\nRunning pairwise alignment ...")

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.5

    best = next(iter(aligner.align(seq1, seq2)))
    counts = best.counts()

    alignment_length = counts.identities + counts.mismatches + counts.gaps
    percent_identity = (counts.identities / alignment_length) * 100

    if verbose:
        print(f"\n--- Results ---")
        print(f"Alignment length    : {alignment_length}")
        print(f"Identical positions : {counts.identities}")
        print(f"Mismatches          : {counts.mismatches}")
        print(f"Gaps                : {counts.gaps}")
        print(f"% Identity          : {percent_identity:.2f}%")

        if percent_identity > 30:
            print(f"Verdict: SIMILAR — likely same protein family / nearly identical 3D structure (>30%)")
        else:
            print(f"Verdict: DIVERGENT — structures likely differ (<30%)")

    return percent_identity


if __name__ == "__main__":
    id1 = "8DNP"
    id2 = "6M54"
    score = similarity_score(id1, id2, verbose=0)
    print(f"\nFinal % Identity between {id1} and {id2}: {score:.2f}%")
