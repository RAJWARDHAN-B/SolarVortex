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
from training.losses import SegmentationObjective
from training.run import coarse_predictions, main


def tiny_model(*args, **kwargs):
    return torch.nn.Conv2d(1, 29, kernel_size=1)


class PipelineTests(unittest.TestCase):
    def test_ce_dice_ignores_pixels_and_has_finite_gradients(self):
        logits = torch.randn(1, 3, 1, 2, requires_grad=True)
        target = torch.tensor([[[1, 255]]])
        objective = SegmentationObjective("ce-dice")
        parts = objective.components(logits, target)
        expected = objective(logits[:, :, :, :1], target[:, :, :1])
        torch.testing.assert_close(parts["total"], expected)
        self.assertGreater(float(parts["soft_dice"]), 0)
        parts["total"].backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        torch.testing.assert_close(logits.grad[:, :, :, 1:], torch.zeros(1, 3, 1, 1))

    def test_ce_dice_background_and_all_ignored_cases(self):
        logits = torch.randn(1, 3, 2, 2, requires_grad=True)
        objective = SegmentationObjective("ce-dice")
        background = torch.zeros(1, 2, 2, dtype=torch.long)
        parts = objective.components(logits, background)
        self.assertEqual(float(parts["soft_dice"]), 0)
        torch.testing.assert_close(parts["total"], torch.nn.functional.cross_entropy(logits, background))
        ignored = torch.full_like(background, 255)
        loss = objective(logits, ignored)
        self.assertEqual(float(loss), 0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

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

    @patch("training.run.build_model", side_effect=tiny_model)
    def test_solar_recipe_preserves_source_loss_and_checkpoint_contract(self, factory):
        arguments = self.arguments("solar-recipe")
        arguments[arguments.index("--model") + 1] = "solar-unet"
        run_dir = main([*arguments, "--variant", "detail", "--loss", "ce-dice"])
        config = json.loads((run_dir / "config.json").read_text())
        self.assertEqual(config["architecture_version"], "solar-unet-v1")
        self.assertEqual(config["loss"], "ce-dice")
        self.assertTrue((run_dir / "source/models/custom/solar_unet.py").is_file())
        self.assertTrue((run_dir / "source/training/losses.py").is_file())
        self.assertTrue((run_dir / "plots/loss_components.png").is_file())
        checkpoint = torch.load(run_dir / "checkpoints/best.pt", weights_only=False)
        self.assertEqual(checkpoint["best_val_loss"], checkpoint["history"][-1]["val"]["cross_entropy"])
        self.assertEqual(checkpoint["history"][-1]["val"]["objective"], "ce-dice")
        evaluation = main(["evaluate", "--run-dir", str(run_dir), "--workers", "0", "--device", "cpu", "--figures", "0"])
        report = json.loads((evaluation / "metrics.json").read_text())
        self.assertEqual(report["objective"], "ce-dice")

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