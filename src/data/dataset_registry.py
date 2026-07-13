from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

DR_CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}


def images_by_stem(folder: Path) -> dict:
    if not folder.exists():
        return {}
    return {p.stem: p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS}


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {c.strip().lower(): c for c in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    return df.columns[0]


def find_label_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "diagnosis",
        "retinopathy grade",
        "retinopathy_grade",
        "grade",
        "label",
    ]

    normalized = {c.strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    return None


def build_image_path_column(df: pd.DataFrame, id_col: str, image_folder: Path) -> pd.DataFrame:
    images = images_by_stem(image_folder)

    image_paths = []
    image_exists = []

    for value in df[id_col].astype(str):
        stem = Path(value).stem
        path = images.get(stem)

        if path is None:
            image_paths.append("")
            image_exists.append(False)
        else:
            image_paths.append(str(path))
            image_exists.append(True)

    df["image_path"] = image_paths
    df["image_exists"] = image_exists
    return df


def load_aptos_split(split: str) -> pd.DataFrame:
    base = RAW / "APTOS2019"

    split_map = {
        "train": (base / "train_1.csv", base / "train_images"),
        "valid": (base / "valid.csv", base / "val_images"),
        "test": (base / "test.csv", base / "test_images"),
    }

    csv_path, image_folder = split_map[split]

    df = pd.read_csv(csv_path)
    id_col = find_column(df, ["id_code", "image", "filename", "file_name"])
    label_col = find_label_column(df)

    if label_col is None:
        raise ValueError(f"No label column found in {csv_path}")

    df = df.copy()
    df["dataset"] = "APTOS2019"
    df["split"] = split
    df["image_id"] = df[id_col].astype(str).apply(lambda x: Path(x).stem)
    df["label"] = df[label_col].astype(int)
    df["label_name"] = df["label"].map(DR_CLASS_NAMES)

    df = build_image_path_column(df, id_col, image_folder)

    return df[["dataset", "split", "image_id", "label", "label_name", "image_path", "image_exists"]]


def load_aptos_all() -> pd.DataFrame:
    return pd.concat(
        [load_aptos_split("train"), load_aptos_split("valid"), load_aptos_split("test")],
        ignore_index=True,
    )


def load_idrid_grading_split(split: str) -> pd.DataFrame:
    base = RAW / "IDRiD" / "disease_grading" / "B. Disease Grading"

    split_map = {
        "train": (
            base / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv",
            base / "1. Original Images" / "a. Training Set",
        ),
        "test": (
            base / "2. Groundtruths" / "b. IDRiD_Disease Grading_Testing Labels.csv",
            base / "1. Original Images" / "b. Testing Set",
        ),
    }

    csv_path, image_folder = split_map[split]

    df = pd.read_csv(csv_path)
    id_col = find_column(df, ["image name", "image", "filename", "file_name"])
    label_col = find_label_column(df)

    if label_col is None:
        raise ValueError(f"No DR grading column found in {csv_path}")

    df = df.copy()
    df["dataset"] = "IDRiD"
    df["split"] = split
    df["image_id"] = df[id_col].astype(str).apply(lambda x: Path(x).stem)
    df["label"] = df[label_col].astype(int)
    df["label_name"] = df["label"].map(DR_CLASS_NAMES)

    df = build_image_path_column(df, id_col, image_folder)

    return df[["dataset", "split", "image_id", "label", "label_name", "image_path", "image_exists"]]


def load_idrid_grading_all() -> pd.DataFrame:
    return pd.concat(
        [load_idrid_grading_split("train"), load_idrid_grading_split("test")],
        ignore_index=True,
    )


def classification_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["dataset", "split", "label", "label_name"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["dataset", "split", "label"])
    )
    return summary


def idrid_segmentation_mask_summary() -> pd.DataFrame:
    base = RAW / "IDRiD" / "segmentation" / "A. Segmentation"

    mask_roots = {
        "train": base / "2. All Segmentation Groundtruths" / "a. Training Set",
        "test": base / "2. All Segmentation Groundtruths" / "b. Testing Set",
    }

    rows = []

    for split, root in mask_roots.items():
        if not root.exists():
            rows.append({
                "dataset": "IDRiD",
                "split": split,
                "mask_folder": "MISSING",
                "count": 0,
                "path": str(root),
            })
            continue

        for folder in sorted([p for p in root.iterdir() if p.is_dir()]):
            count = sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
            rows.append({
                "dataset": "IDRiD",
                "split": split,
                "mask_folder": folder.name,
                "count": count,
                "path": str(folder),
            })

    return pd.DataFrame(rows)
