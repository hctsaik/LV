import numpy as np
import pytest
from pathlib import Path


def test_parse_label_single_class(tmp_path):
    from visualize_embeddings import parse_label_file

    (tmp_path / "img.txt").write_text("0 0.5 0.5 0.3 0.4\n")
    assert parse_label_file(tmp_path / "img.txt", ["apple", "banana", "orange"]) == "apple"


def test_parse_label_class_id_1(tmp_path):
    from visualize_embeddings import parse_label_file

    (tmp_path / "img.txt").write_text("1 0.5 0.5 0.3 0.4\n")
    assert parse_label_file(tmp_path / "img.txt", ["apple", "banana", "orange"]) == "banana"


def test_parse_label_multi_class_is_mix(tmp_path):
    from visualize_embeddings import parse_label_file

    (tmp_path / "img.txt").write_text("0 0.5 0.5 0.3 0.4\n1 0.2 0.3 0.1 0.1\n")
    assert parse_label_file(tmp_path / "img.txt", ["apple", "banana"]) == "mix"


def test_parse_label_missing_file_is_unknown(tmp_path):
    from visualize_embeddings import parse_label_file

    assert parse_label_file(tmp_path / "missing.txt", ["apple"]) == "unknown"


def test_parse_label_empty_file_is_unknown(tmp_path):
    from visualize_embeddings import parse_label_file

    (tmp_path / "empty.txt").write_text("")
    assert parse_label_file(tmp_path / "empty.txt", ["apple"]) == "unknown"


def test_discover_images_finds_multiple_splits(tmp_path):
    from visualize_embeddings import discover_images

    for split in ("train", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
        (tmp_path / split / "images" / "apple_1.jpg").write_bytes(b"fake")
        (tmp_path / split / "labels" / "apple_1.txt").write_text("0 0.5 0.5 0.3 0.4\n")

    records = discover_images(tmp_path, ["apple", "banana"])
    assert len(records) == 2
    assert {r["split"] for r in records} == {"train", "test"}


def test_discover_images_skips_absent_split(tmp_path):
    from visualize_embeddings import discover_images

    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "labels").mkdir(parents=True)
    (tmp_path / "train" / "images" / "x.jpg").write_bytes(b"fake")
    (tmp_path / "train" / "labels" / "x.txt").write_text("0 0.5 0.5 0.3 0.4\n")

    records = discover_images(tmp_path, ["apple"])
    assert all(r["split"] == "train" for r in records)


def test_discover_images_label_assigned(tmp_path):
    from visualize_embeddings import discover_images

    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "labels").mkdir(parents=True)
    (tmp_path / "train" / "images" / "img.jpg").write_bytes(b"fake")
    (tmp_path / "train" / "labels" / "img.txt").write_text("2 0.5 0.5 0.3 0.4\n")

    records = discover_images(tmp_path, ["apple", "banana", "orange"])
    assert records[0]["label"] == "orange"
