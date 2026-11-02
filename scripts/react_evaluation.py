"""Request-time adapter for the legacy consensus-set detection evaluation."""
from __future__ import annotations

import csv
import io
from collections import Counter
from pathlib import Path

from _utils import yaml_class_names
from evaluation import consensus_flags, evaluate_detections
from interaction import parse_predictions_csv_text, parse_yolo_boxes, yolo_label_path_for
from safe_io import safe_read_text


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class EvaluationInputError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _read_classes_txt(folder: Path) -> list[str] | None:
    classes_file = folder.parent / "classes.txt"
    if classes_file.exists():
        names = [line.strip() for line in safe_read_text(classes_file).splitlines() if line.strip()]
        if names:
            return names
    return yaml_class_names(folder.parent / "data.yaml")


def _classes_txt_nested(folder: Path) -> list[str] | None:
    for candidate in (folder, folder / "_", folder.parent):
        names = _read_classes_txt(candidate)
        if names:
            return names
    return None


def _gt_by_image(folder: Path, names: list[str]) -> dict[str, list[dict]]:
    def class_name(class_id: int) -> str:
        return names[class_id] if 0 <= class_id < len(names) else f"class_{class_id}"

    gt: dict[str, list[dict]] = {}
    for image_path in sorted((folder / "images").rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        boxes = parse_yolo_boxes(yolo_label_path_for(image_path))
        if boxes:
            gt[image_path.name] = [
                {"cls": class_name(class_id), "cx": cx, "cy": cy, "w": width, "h": height}
                for class_id, cx, cy, width, height in boxes
            ]
    return gt


def _gt_for_folders(
    folders: list[Path],
) -> tuple[dict[str, list[dict]], dict[str, str], list[str]]:
    combined: dict[str, list[dict]] = {}
    image_by_name: dict[str, str] = {}
    duplicates: set[str] = set()
    for folder in folders:
        names = _classes_txt_nested(folder) or []
        root_gt = _gt_by_image(folder, names)
        for image_path in sorted((folder / "images").rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if image_path.name in image_by_name and image_by_name[image_path.name] != str(image_path):
                duplicates.add(image_path.name)
            else:
                image_by_name[image_path.name] = str(image_path)
        for filename, boxes in root_gt.items():
            if filename not in duplicates:
                combined[filename] = boxes
    for filename in duplicates:
        combined.pop(filename, None)
        image_by_name.pop(filename, None)
    return combined, image_by_name, sorted(duplicates)


def run_detection_evaluation(
    folders: list[str],
    prediction_csv: str,
    consensus_csv: str | None,
    *,
    iou_thresh: float,
    conf_thresh: float,
    class_aware: bool,
) -> dict:
    if not folders:
        raise EvaluationInputError("DATASET_REQUIRED", "Select at least one image dataset folder.")

    resolved_folders: list[Path] = []
    for raw_folder in folders:
        try:
            folder = Path(raw_folder).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise EvaluationInputError(
                "DATASET_UNAVAILABLE", "Each selected dataset must contain an images folder.",
            ) from exc
        if not folder.is_dir() or not (folder / "images").is_dir():
            raise EvaluationInputError(
                "DATASET_UNAVAILABLE", "Each selected dataset must contain an images folder.",
            )
        if folder not in resolved_folders:
            resolved_folders.append(folder)

    gt_by_image, image_by_name, duplicate_names = _gt_for_folders(resolved_folders)
    if duplicate_names:
        names = ", ".join(duplicate_names[:20])
        raise EvaluationInputError(
            "DUPLICATE_IMAGE_FILENAMES",
            f"Image filenames must be unique across selected folders. Rename duplicates: {names}",
        )
    if not gt_by_image:
        raise EvaluationInputError("GROUND_TRUTH_REQUIRED", "No YOLO ground-truth boxes were found in labels/.")

    try:
        predictions = parse_predictions_csv_text(prediction_csv)
        consensus_by_image = None
        consensus_gt_count = None
        gray_gt_count = None
        if consensus_csv is not None:
            rows = list(csv.DictReader(io.StringIO(consensus_csv)))
            consensus_by_image, consensus_gt_count, gray_gt_count = consensus_flags(rows, gt_by_image)
    except (csv.Error, UnicodeError, ValueError, TypeError) as exc:
        raise EvaluationInputError("INVALID_CSV", "A CSV file could not be read. Check its header and encoding.") from exc

    result = evaluate_detections(
        gt_by_image,
        predictions,
        iou_thresh=iou_thresh,
        conf_thresh=conf_thresh,
        class_aware=class_aware,
        consensus_by_image=consensus_by_image,
    )
    gt_classes = {str(box["cls"]) for boxes in gt_by_image.values() for box in boxes}
    prediction_classes = {str(box["cls"]) for boxes in predictions.values() for box in boxes}
    warnings: list[str] = []
    if not predictions:
        warnings.append("NO_PREDICTIONS_MATCHED")
    if class_aware and gt_classes and prediction_classes and not (gt_classes & prediction_classes):
        warnings.append("CLASS_NAMESPACE_MISMATCH")
    if consensus_by_image is None:
        warnings.append("CONSENSUS_NOT_PROVIDED")

    per_class = [
        {"class_name": class_name, **values}
        for class_name, values in sorted(
            result["per_class"].items(), key=lambda pair: (-pair[1]["fn"], pair[0]),
        )
    ]
    confusion_counts = Counter(
        (ground_truth, predicted)
        for ground_truth, predicted in result["confusion"]
        if ground_truth and predicted and ground_truth != predicted
    )
    confusion = [
        {"ground_truth": pair[0], "predicted": pair[1], "count": count}
        for pair, count in confusion_counts.most_common(8)
    ]
    escapes = result["false_negatives"]
    public_escapes = [
        {"index": index, "filename": item["filename"], "class_name": item["cls"], "box": item["box"]}
        for index, item in enumerate(escapes[:40])
    ]
    return {
        "summary": {
            "overall": result["overall"],
            "per_class": per_class,
            "escape_count": len(escapes),
            "escapes": public_escapes,
            "confusion": confusion,
            "gray_gt_count": result["gray"]["total"],
            "consensus_gt_count": consensus_gt_count,
            "image_count": result["n_images"],
            "prediction_image_count": len(predictions),
            "used_consensus": consensus_by_image is not None,
            "warnings": warnings,
        },
        "image_by_name": image_by_name,
        "escapes": escapes,
    }


def run_detection_evaluation_demo(sample_root: Path) -> dict:
    """Recreate Streamlit's explicit sample journey without implying model output."""
    demo_folder = Path(__file__).resolve().parents[1] / "demo" / "coco8" / "train"
    if not (demo_folder / "images").is_dir():
        from sample_data import ensure_detection_sample

        demo_folder = ensure_detection_sample(Path(sample_root) / "detection-evaluation")
    gt_by_image, _image_by_name, duplicate_names = _gt_for_folders([demo_folder])
    if duplicate_names:
        raise EvaluationInputError("DUPLICATE_IMAGE_FILENAMES", "The bundled sample contains duplicate image names.")
    if not gt_by_image:
        raise EvaluationInputError("GROUND_TRUTH_REQUIRED", "The evaluation sample contains no YOLO boxes.")

    filenames = list(gt_by_image)
    missed_filename = filenames[-1]
    gray_filename = filenames[-2] if len(filenames) >= 2 else None
    prediction_csv = io.StringIO()
    prediction_writer = csv.writer(prediction_csv)
    prediction_writer.writerow(["filename", "class", "cx", "cy", "w", "h", "score"])
    consensus_csv = io.StringIO()
    consensus_writer = csv.writer(consensus_csv)
    consensus_writer.writerow(["filename", "consensus", "cx", "cy", "w", "h"])
    for filename, boxes in gt_by_image.items():
        for box in boxes:
            if filename != missed_filename:
                prediction_writer.writerow([
                    filename, box["cls"], box["cx"], box["cy"], box["w"], box["h"], 0.9,
                ])
            consensus_writer.writerow([
                filename, filename != gray_filename,
                box["cx"], box["cy"], box["w"], box["h"],
            ])

    return run_detection_evaluation(
        [str(demo_folder)],
        prediction_csv.getvalue(),
        consensus_csv.getvalue(),
        iou_thresh=0.5,
        conf_thresh=0.0,
        class_aware=True,
    )
