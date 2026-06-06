from pathlib import Path
import pytest
from app import parse_folder_paths, read_classes_txt


def test_read_classes_txt_found(tmp_path):
    folder = tmp_path / "train"
    folder.mkdir()
    (tmp_path / "classes.txt").write_text("apple\nbanana\norange\n")
    assert read_classes_txt(folder) == ["apple", "banana", "orange"]


def test_read_classes_txt_not_found(tmp_path):
    folder = tmp_path / "train"
    folder.mkdir()
    assert read_classes_txt(folder) is None


def test_read_classes_txt_empty_file(tmp_path):
    folder = tmp_path / "train"
    folder.mkdir()
    (tmp_path / "classes.txt").write_text("\n\n")
    assert read_classes_txt(folder) is None


def test_parse_folder_paths_basic(tmp_path):
    text = f"{tmp_path}/train\n{tmp_path}/test"
    result = parse_folder_paths(text)
    assert result == [Path(f"{tmp_path}/train"), Path(f"{tmp_path}/test")]


def test_parse_folder_paths_ignores_blank_lines(tmp_path):
    text = f"{tmp_path}/train\n\n  \n{tmp_path}/test"
    result = parse_folder_paths(text)
    assert len(result) == 2


def test_parse_folder_paths_empty():
    assert parse_folder_paths("") == []
    assert parse_folder_paths("   \n  ") == []
