from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports" / "tables"
REPORTS.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def list_images_by_stem(folder: Path) -> dict:
    if not folder.exists():
        return {}
    return {p.stem: p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS}


def add_row(rows, dataset, component, split, item_type, path, count="", status="OK", details=""):
    rows.append({
        "dataset": dataset,
        "component": component,
        "split": split,
        "item_type": item_type,
        "path": str(path),
        "path_exists": path.exists(),
        "count": count,
        "status": status,
        "details": details,
    })


def check_csv_with_images(rows, dataset, component, split, csv_path: Path, image_folder: Path):
    if not csv_path.exists():
        add_row(rows, dataset, component, split, "csv", csv_path, 0, "MISSING", "CSV file not found")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        add_row(rows, dataset, component, split, "csv", csv_path, 0, "ERROR", f"Could not read CSV: {e}")
        return

    add_row(rows, dataset, component, split, "csv_rows", csv_path, len(df), "OK", f"columns={list(df.columns)}")

    if not image_folder.exists():
        add_row(rows, dataset, component, split, "image_folder", image_folder, 0, "MISSING", "Image folder not found")
        return

    images = list_images_by_stem(image_folder)
    add_row(rows, dataset, component, split, "images", image_folder, len(images), "OK", "Images counted by file stem")

    id_col = None
    for candidate in ["id_code", "image", "Image name", "Image", "filename", "file_name"]:
        if candidate in df.columns:
            id_col = candidate
            break

    if id_col is None and len(df.columns) > 0:
        id_col = df.columns[0]

    if id_col is not None:
        missing = []
        for value in df[id_col].astype(str):
            stem = Path(value).stem
            if stem not in images:
                missing.append(value)

        status = "OK" if len(missing) == 0 else "WARNING"
        add_row(
            rows,
            dataset,
            component,
            split,
            "csv_image_match",
            image_folder,
            f"missing={len(missing)}",
            status,
            f"id_col={id_col}; first_missing={missing[:5]}"
        )

    label_col = None
    for candidate in ["diagnosis", "Retinopathy grade", "retinopathy_grade", "grade", "label"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is not None:
        dist = df[label_col].value_counts().sort_index().to_dict()
        add_row(rows, dataset, component, split, "class_distribution", csv_path, len(dist), "OK", str(dist))


def verify_aptos(rows):
    aptos = RAW / "APTOS2019"

    add_row(rows, "APTOS2019", "root", "all", "folder", aptos, "", "OK" if aptos.exists() else "MISSING")

    checks = [
        ("train", aptos / "train_1.csv", aptos / "train_images"),
        ("valid", aptos / "valid.csv", aptos / "val_images"),
        ("test", aptos / "test.csv", aptos / "test_images"),
    ]

    for split, csv_path, img_folder in checks:
        check_csv_with_images(rows, "APTOS2019", "disease_grading", split, csv_path, img_folder)


def verify_idrid_grading(rows):
    base = RAW / "IDRiD" / "disease_grading" / "B. Disease Grading"

    train_img = base / "1. Original Images" / "a. Training Set"
    test_img = base / "1. Original Images" / "b. Testing Set"

    train_csv = base / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv"
    test_csv = base / "2. Groundtruths" / "b. IDRiD_Disease Grading_Testing Labels.csv"

    add_row(rows, "IDRiD", "disease_grading", "root", "folder", base, "", "OK" if base.exists() else "MISSING")

    check_csv_with_images(rows, "IDRiD", "disease_grading", "train", train_csv, train_img)
    check_csv_with_images(rows, "IDRiD", "disease_grading", "test", test_csv, test_img)


def verify_idrid_segmentation(rows):
    base = RAW / "IDRiD" / "segmentation" / "A. Segmentation"

    train_img = base / "1. Original Images" / "a. Training Set"
    test_img = base / "1. Original Images" / "b. Testing Set"

    train_masks = base / "2. All Segmentation Groundtruths" / "a. Training Set"
    test_masks = base / "2. All Segmentation Groundtruths" / "b. Testing Set"

    add_row(rows, "IDRiD", "segmentation", "root", "folder", base, "", "OK" if base.exists() else "MISSING")
    add_row(rows, "IDRiD", "segmentation", "train", "original_images", train_img, count_images(train_img), "OK" if train_img.exists() else "MISSING")
    add_row(rows, "IDRiD", "segmentation", "test", "original_images", test_img, count_images(test_img), "OK" if test_img.exists() else "MISSING")

    for split, mask_root in [("train", train_masks), ("test", test_masks)]:
        add_row(rows, "IDRiD", "segmentation", split, "mask_root", mask_root, "", "OK" if mask_root.exists() else "MISSING")
        if mask_root.exists():
            for sub in sorted([p for p in mask_root.iterdir() if p.is_dir()]):
                add_row(
                    rows,
                    "IDRiD",
                    "segmentation",
                    split,
                    "mask_folder",
                    sub,
                    count_images(sub),
                    "OK",
                    f"folder={sub.name}"
                )


def verify_idrid_localization(rows):
    base = RAW / "IDRiD" / "localization" / "C. Localization"

    train_img = base / "1. Original Images" / "a. Training Set"
    test_img = base / "1. Original Images" / "b. Testing Set"
    gt = base / "2. Groundtruths"

    add_row(rows, "IDRiD", "localization", "root", "folder", base, "", "OK" if base.exists() else "MISSING")
    add_row(rows, "IDRiD", "localization", "train", "original_images", train_img, count_images(train_img), "OK" if train_img.exists() else "MISSING")
    add_row(rows, "IDRiD", "localization", "test", "original_images", test_img, count_images(test_img), "OK" if test_img.exists() else "MISSING")

    if gt.exists():
        csv_files = list(gt.rglob("*.csv"))
        add_row(rows, "IDRiD", "localization", "all", "csv_files", gt, len(csv_files), "OK", str([p.name for p in csv_files]))
    else:
        add_row(rows, "IDRiD", "localization", "all", "groundtruth_folder", gt, 0, "MISSING", "Groundtruth folder not found")


def main():
    rows = []

    print("=" * 80)
    print("RetinaGuard-AI Dataset Verification")
    print("=" * 80)
    print(f"Project root: {ROOT}")
    print(f"Raw data path: {RAW}")
    print()

    verify_aptos(rows)
    verify_idrid_grading(rows)
    verify_idrid_segmentation(rows)
    verify_idrid_localization(rows)

    df = pd.DataFrame(rows)
    out_path = REPORTS / "dataset_summary.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(df[["dataset", "component", "split", "item_type", "count", "status"]].to_string(index=False))
    print()
    print("=" * 80)
    print(f"Saved report to: {out_path}")
    print("=" * 80)

    problems = df[df["status"].isin(["MISSING", "ERROR"])]
    if len(problems) > 0:
        print()
        print("WARNING: Some paths/files are missing or unreadable:")
        print(problems[["dataset", "component", "split", "item_type", "path", "status"]].to_string(index=False))
    else:
        print("All critical dataset checks passed.")


if __name__ == "__main__":
    main()
