import cv2
import numpy as np
import os


def isolate_protein(input_path: str, output_path: str,
                    blur_ksize: int = 5,
                    close_ksize: int = 5,
                    dilate_px: int = 2) -> None:
    """
    Replace the noisy background of a protein thumbnail with pure black,
    keeping only the central protein blob.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) Smooth noise so background pixels cluster tightly
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # 2) Otsu picks an automatic threshold separating background vs foreground
    _, mask = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3) Close small holes inside the protein blob, then open to drop specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (close_ksize, close_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 4) Keep only the largest connected component (the protein, not text/noise)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8) * 255

    # 5) Slight dilation so we don't shave the protein's edge
    if dilate_px > 0:
        d_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px*2+1, dilate_px*2+1))
        mask = cv2.dilate(mask, d_kernel)

    # 6) Apply mask: protein pixels stay, everything else becomes black
    out = cv2.bitwise_and(img, img, mask=mask)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, out)


def clean_all(root: str) -> None:
    """
    Walk every subfolder of `root` (e.g. 7UZM, 8DNM, ...) and replace each
    PNG/JPG with its background-cleaned version, in-place.
    """
    exts = {".png", ".jpg", ".jpeg"}
    total = 0

    for sub in sorted(os.listdir(root)):
        sub_path = os.path.join(root, sub)
        if not os.path.isdir(sub_path):
            continue

        files = sorted(
            f for f in os.listdir(sub_path)
            if os.path.splitext(f)[1].lower() in exts
        )
        for fname in files:
            path = os.path.join(sub_path, fname)
            isolate_protein(path, path)

        print(f"  [{sub}] cleaned {len(files)} images")
        total += len(files)

    print(f"\nDone: cleaned {total} images under {root}")


if __name__ == "__main__":
    root = "/data/atran16/ProteinClassification_3D/3D_PDB_5013/newProfessorSuData"
    clean_all(root)