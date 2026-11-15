import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from datasets.el_classes import EL_CLASSES
from training.data import load_datasets, prepare_manifest
from training.run import coarse_predictions, main


def tiny_model(*args, **kwargs):
    return torch.nn.Conv2d(1, 29, kernel_size=1)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        for index, split in enumerate(("train", "val", "test")):
            for kind in ("images", "masks"):
                directory = self.data / f"el_{kind}_{split}"
                directory.mkdir(parents=True)
                pixels = np.full((32, 32), 110 if kind == "images" else index, dtype=np.uint8)
                Image.fromarray(pixels).save(directory / f"ARTS_{index:05d}_r1_c1.png")
        with (self.data / "ListOfClassesAndColorCodes_test.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["Label", "Desc", "Class", "Red", "Green", "Blue"])
            writer.writeheader()
            for entry in EL_CLASSES:
                writer.writerow(dict(zip(writer.fieldnames, (entry.label, entry.name, entry.kind, *entry.rgb))))

    def arguments(self, name):
        return ["train", "--model", "unet", "--data-root", str(self.data), "--output-root", str(self.root / "runs"), "--run-name", name, "--epochs", "1", "--batch-size", "1", "--workers", "0", "--device", "cpu"]

    @patch("training.run.build_model", side_effect=tiny_model)
    def test_training_evaluation_resume_and_no_overwrite(self, factory):
        run_dir = main(self.arguments("integration"))
        self.assertTrue((run_dir / "checkpoints/best.pt").is_file())
        self.assertTrue((run_dir / "source/training/run.py").is_file())
        history = json.loads((run_dir / "history.json").read_text())
        self.assertEqual(len(history), 1)
        json.dumps(history, allow_nan=False)
        with Image.open(run_dir / "plots/training_curves.png") as figure:
            self.assertGreater(np.asarray(figure).std(), 0)
        evaluation = main(["evaluate", "--run-dir", str(run_dir), "--workers", "0", "--device", "cpu", "--figures", "1"])
        metrics = json.loads((evaluation / "metrics.json").read_text())
        self.assertEqual(metrics["split"], "test")
        self.assertEqual(metrics["el29"]["valid_pixels"], 32 * 32)
        prediction = next((evaluation / "predictions").glob("*.png"))
        with Image.open(prediction) as image:
            self.assertEqual(image.size, (32, 32))
            self.assertEqual(image.mode, "L")
        self.assertEqual(len(list((evaluation / "qualitative").glob("*.png"))), 1)
        self.assertEqual(np.load(evaluation / "el29_confusion_counts.npy").shape, (29, 29))
        with self.assertRaises(FileExistsError):
            main(self.arguments("integration"))
        main(["train", "--resume", str(run_dir / "checkpoints/last.pt"), "--epochs", "2", "--workers", "0", "--device", "cpu"])
        self.assertEqual(len(json.loads((run_dir / "history.json").read_text())), 2)
        resumed = torch.load(run_dir / "checkpoints/last.pt", weights_only=False)
        direct_args = self.arguments("uninterrupted")
        direct_args[direct_args.index("--epochs") + 1] = "2"
        direct_dir = main(direct_args)
        direct = torch.load(direct_dir / "checkpoints/last.pt", weights_only=False)
        for name in direct["model"]:
            torch.testing.assert_close(resumed["model"][name], direct["model"][name], rtol=0, atol=0)
        mask = next((self.data / "el_masks_test").glob("*.png"))
        Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(mask)
        with self.assertRaisesRegex(ValueError, "Dataset content differs"):
            main(["evaluate", "--run-dir", str(run_dir), "--workers", "0", "--device", "cpu"])

    def test_manifest_checksum(self):
        manifest = prepare_manifest(self.data, "module_grouped", 42)
        manifest["samples"]["test"][0]["image"] = "wrong.png"
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_datasets(self.data, manifest)

    def test_orphan_mask_is_rejected(self):
        Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(self.data / "el_masks_train/orphan.png")
        with self.assertRaisesRegex(ValueError, "Unpaired files"):
            prepare_manifest(self.data, "module_grouped", 42)

    @patch("training.run.build_model", side_effect=tiny_model)
    def test_smoke_is_separate_from_benchmark_runs(self, factory):
        directory = main(["smoke", "--model", "unet", "--data-root", str(self.data), "--output-root", str(self.root / "runs"), "--workers", "0", "--device", "cpu"])
        self.assertTrue((directory / "smoke.json").is_file())
        self.assertFalse((directory / "checkpoints").exists())

    def test_legacy_cleanup_preserves_weights(self):
        legacy = self.root / "results/U-Net"
        legacy.mkdir(parents=True)
        (legacy / "best_unet.pth").write_bytes(b"preserve")
        main(["cleanup", "--output-root", str(self.root / "results/runs")])
        archived = list((self.root / "results/legacy").rglob("best_unet.pth"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_bytes(), b"preserve")
        self.assertFalse(legacy.exists())

    def test_coarse_probability_aggregation(self):
        logits = torch.full((1, 29, 1, 1), -20.0)
        logits[:, 14] = 2.0
        logits[:, 1] = 1.6
        logits[:, 2] = 1.6
        self.assertEqual(int(logits.argmax(1).item()), 14)
        self.assertEqual(int(coarse_predictions(logits).item()), 1)


if __name__ == "__main__":
    unittest.main()