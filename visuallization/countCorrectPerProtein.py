import pandas as pd


def count_per_protein(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    # Normalize hit column to bool (CSV stores it as the string "True"/"False")
    df["hit"] = df["hit"].astype(str).str.strip().str.lower() == "true"

    grouped = (
        df.groupby("ground_truth")
          .agg(correct=("hit", "sum"), total=("hit", "size"))
          .sort_index()
    )

    total_correct = int(grouped["correct"].sum())
    total_count   = int(grouped["total"].sum())

    print(f"Per-protein accuracy ({len(grouped)} proteins):\n")
    for protein, row in grouped.iterrows():
        c, t = int(row["correct"]), int(row["total"])
        print(f"  Protein {protein}: {c}/{t}  ({c/t:.1%})")

    print(f"\nOverall: {total_correct}/{total_count}  "
          f"({total_correct/total_count:.1%})")


if __name__ == "__main__":
    csv_path = "/data/atran16/ProteinClassification_3D/visuallization/testVisuallizaion/SequentialSimilarityCheckResult.csv"
    count_per_protein(csv_path)