"""Tests for dataset loading and the bundled TextVQA sample."""

from __future__ import annotations

import json

import pytest

from vea.data import DATA_ROOT, VQASample, available_datasets, load_samples


class TestLoadSamples:
    def test_bundled_textvqa_loads(self):
        samples = load_samples("textvqa")
        assert samples
        assert all(isinstance(s, VQASample) for s in samples)

    def test_every_referenced_image_exists(self):
        # The metadata and the image directory must stay in sync; a missing file
        # would only surface deep inside a run otherwise.
        missing = [s.image_path.name for s in load_samples("textvqa") if not s.image_path.is_file()]
        assert not missing, f"missing images: {missing}"

    def test_images_load_as_rgb(self):
        image = load_samples("textvqa", limit=1)[0].load_image()
        assert image.mode == "RGB"

    def test_boxes_lie_inside_the_declared_image_size(self):
        metadata = json.loads((DATA_ROOT / "textvqa" / "metadata.json").read_text())
        for index, record in enumerate(metadata):
            width, height = record["width"], record["height"]
            for x0, y0, x1, y1 in record["bboxs"]:
                assert 0 <= x0 < x1 <= width, f"record {index}: x range {x0}-{x1} vs {width}"
                assert 0 <= y0 < y1 <= height, f"record {index}: y range {y0}-{y1} vs {height}"

    def test_declared_size_matches_the_actual_image(self):
        metadata = json.loads((DATA_ROOT / "textvqa" / "metadata.json").read_text())
        for record, sample in zip(metadata, load_samples("textvqa"), strict=True):
            image = sample.load_image()
            assert image.size == (record["width"], record["height"]), sample.image_path.name

    def test_every_sample_has_a_question_answer_and_box(self):
        for sample in load_samples("textvqa"):
            assert sample.question.strip()
            assert sample.answers and sample.answers[0].strip()
            assert sample.boxes

    def test_limit_truncates(self):
        assert len(load_samples("textvqa", limit=3)) == 3

    def test_sample_ids_are_positional(self):
        samples = load_samples("textvqa", limit=5)
        assert [s.sample_id for s in samples] == [0, 1, 2, 3, 4]

    def test_missing_dataset_reports_what_is_available(self):
        with pytest.raises(FileNotFoundError, match="Datasets present"):
            load_samples("nonexistent-dataset")


class TestAvailableDatasets:
    def test_finds_the_bundled_dataset(self):
        assert "textvqa" in available_datasets()

    def test_missing_root_yields_nothing(self, tmp_path):
        assert available_datasets(tmp_path / "absent") == []


class TestSchemaFlexibility:
    def _write(self, tmp_path, record):
        dataset_dir = tmp_path / "toy"
        (dataset_dir / "images").mkdir(parents=True)
        from PIL import Image

        Image.new("RGB", (8, 8)).save(dataset_dir / "images" / "a.jpg")
        (dataset_dir / "metadata.json").write_text(json.dumps([record]))
        return tmp_path

    def test_accepts_an_answers_list(self, tmp_path):
        root = self._write(tmp_path, {"question": "q", "answers": ["x", "y"], "image": "a.jpg"})
        assert load_samples("toy", root=root)[0].answers == ["x", "y"]

    def test_accepts_a_single_answer(self, tmp_path):
        root = self._write(tmp_path, {"question": "q", "answer": "x", "image": "a.jpg"})
        assert load_samples("toy", root=root)[0].answers == ["x"]

    def test_boxes_are_optional(self, tmp_path):
        root = self._write(tmp_path, {"question": "q", "answer": "x", "image": "a.jpg"})
        assert load_samples("toy", root=root)[0].boxes == []

    def test_missing_answer_is_rejected(self, tmp_path):
        root = self._write(tmp_path, {"question": "q", "image": "a.jpg"})
        with pytest.raises(ValueError, match="has no answer"):
            load_samples("toy", root=root)

    def test_missing_required_field_is_rejected(self, tmp_path):
        root = self._write(tmp_path, {"answer": "x", "image": "a.jpg"})
        with pytest.raises(ValueError, match=r"missing \['question'\]"):
            load_samples("toy", root=root)
