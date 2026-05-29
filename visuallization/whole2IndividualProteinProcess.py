import cv2
import numpy as np
import os


def split_protein_sheet(input_path: str, output_dir: str, index_start: int) -> int:
    """
    Split a sheet image containing multiple protein thumbnails separated by
    white/bright borders into individual numbered images.

    Returns the number of cells saved.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # A row/column is a separator when every pixel in it is bright (min >= threshold).
    # Using min instead of mean avoids misclassifying rows that have dark protein
    # pixels on one side but white empty space on the other.
    threshold = 200

    def find_segments(pixel_min, size):
        """Return (start, end) slices for non-separator (content) segments."""
        is_sep = pixel_min >= threshold
        segments = []
        in_seg = False
        start = 0
        for i in range(size):
            if not is_sep[i]:
                if not in_seg:
                    start = i
                    in_seg = True
            else:
                if in_seg:
                    segments.append((start, i))
                    in_seg = False
        if in_seg:
            segments.append((start, size))
        return segments

    row_segs = find_segments(gray.min(axis=1), h)
    col_segs = find_segments(gray.min(axis=0), w)

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for r_start, r_end in row_segs:
        for c_start, c_end in col_segs:
            cell_gray = gray[r_start:r_end, c_start:c_end]

            # Skip tiny artefacts (< 20px on either side)
            if cell_gray.shape[0] < 20 or cell_gray.shape[1] < 20:
                continue

            # Skip cells that are entirely white/blank (no actual protein content)
            if cell_gray.min() >= threshold:
                continue

            cell = img[r_start:r_end, c_start:c_end]
            count += 1
            out_path = os.path.join(output_dir, f"{count+index_start}.png")
            cv2.imwrite(out_path, cell)
            print(f"  Saved [{count}] ({cell.shape[1]}x{cell.shape[0]}) -> {out_path}")

    print(f"\nTotal cells saved: {count}")
    return count


if __name__ == "__main__":
    import argparse

    input = "/data/atran16/ProteinClassification_3D/3D_PDB_5013/newProfessorSuData/8EOR.png"
    output = "/data/atran16/ProteinClassification_3D/3D_PDB_5013/newProfessorSuData/8EOR"
    split_protein_sheet(input, output, index_start = 0)