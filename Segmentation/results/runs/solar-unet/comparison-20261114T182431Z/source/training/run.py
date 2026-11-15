from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.el_classes import CLASS_NAMES, DEFECT_LABELS, EL_CLASSES, EL10_GROUPS, EL10_LUT, EL10_NAMES, NUM_CLASSES
from evaluation.metrics import confusion_matrix, summarize_confusion
from evaluation.reporting import plot_history, save_evaluation, save_prediction, write_csv, write_json
from training.data import audit_datasets, load_datasets, prepare_manifest
from training.losses import SegmentationObjective
from training.models import MODEL_NAMES, build_model

ROOT = Path(__file__).resolve().parents[1]


def timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def environment(device):
    versions = {}
    for package in ("torch", "torchvision", "numpy", "pillow", "matplotlib", "segmentation-models-pytorch", "timm", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "packages": versions, "device": str(device), "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "command": sys.argv, "created_utc": timestamp(),
        "reproducibility": "Seeded Python, NumPy, PyTorch and workers; cuDNN deterministic enabled. Bitwise equality across hardware/software versions is not guaranteed.",
    }


def coarse_predictions(logits):
    probabilities = logits.detach().softmax(dim=1)
    grouped = torch.stack([probabilities[:, list(EL10_GROUPS[name])].sum(dim=1) for name in EL10_NAMES], dim=1)
    return grouped.argmax(dim=1)


def make_loader(dataset, batch_size, workers, device, shuffle=False, seed=42):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def run_epoch(model, loader, device, optimizer=None, output_dir=None, figures=12, objective=None):
    training = optimizer is not None
    model.train(training)
    native = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    coarse = np.zeros((len(EL10_NAMES), len(EL10_NAMES)), dtype=np.int64)
    lookup = torch.tensor(EL10_LUT, device=device, dtype=torch.long)
    loss_sum, cross_entropy_sum, dice_sum, valid_pixels, sample_index = 0.0, 0.0, 0.0, 0, 0
    start = time.perf_counter()
    objective = objective if objective is not None else SegmentationObjective()
    per_image = []
    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            expected = (images.shape[0], NUM_CLASSES, *masks.shape[-2:])
            if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected:
                raise ValueError(f"Expected logits {expected}, got {getattr(logits, 'shape', type(logits))}")
            count = int((masks != 255).sum().item())
            if count == 0:
                raise ValueError("Batch contains only ignored pixels")
            components = objective.components(logits, masks)
            loss = components["total"]
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss; training stopped")
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.detach()) * count
            cross_entropy_sum += float(components["cross_entropy"].detach()) * count
            dice_sum += float(components["soft_dice"].detach()) * count
            valid_pixels += count
            with torch.no_grad():
                predictions = logits.detach().argmax(dim=1)
                coarse_pred = coarse_predictions(logits)
                coarse_target = torch.full_like(masks, 255)
                valid = masks != 255
                coarse_target[valid] = lookup[masks[valid]]
                native += confusion_matrix(predictions, masks, NUM_CLASSES)
                coarse += confusion_matrix(coarse_pred, coarse_target, len(EL10_NAMES))
                if output_dir is not None:
                    for offset in range(images.shape[0]):
                        name = loader.dataset.samples[sample_index][0].name
                        image_report = summarize_confusion(confusion_matrix(predictions[offset], masks[offset], NUM_CLASSES), CLASS_NAMES, DEFECT_LABELS)
                        per_image.append({"image": name, "macro_iou": image_report["macro_iou"], "macro_dice": image_report["macro_dice"], "defect_macro_iou": image_report["defect_macro_iou"]})
                        save_prediction(
                            output_dir, name, images[offset, 0].cpu().numpy(), masks[offset].cpu().numpy(),
                            predictions[offset].cpu().numpy(), sample_index < figures,
                        )
                        sample_index += 1
    if not valid_pixels:
        raise ValueError("No valid pixels evaluated")
    if output_dir is not None:
        write_csv(Path(output_dir) / "per_image.csv", per_image)
        write_json(Path(output_dir) / "per_image.json", per_image)
    return {
        "loss": loss_sum / valid_pixels, "seconds": time.perf_counter() - start,
        "objective": objective.name,
        "cross_entropy": cross_entropy_sum / valid_pixels,
        "soft_dice": dice_sum / valid_pixels,
        "el29": summarize_confusion(native, CLASS_NAMES, DEFECT_LABELS),
        "el10": summarize_confusion(coarse, EL10_NAMES, range(4, 10)),
        "el10_protocol": "Derived from summed EL-29 softmax probabilities per EL-10 group, then argmax. Ground-truth meas_artifact (23) is ignored only in EL-10. This is NOT a separately trained 10-class model.",
    }


def save_history(run_dir, history):
    write_json(run_dir / "history.json", history)
    rows = []
    for entry in history:
        row = {"epoch": entry["epoch"], "learning_rate": entry["learning_rate"]}
        for phase in ("train", "val"):
            row[f"{phase}_loss"] = entry[phase]["loss"]
            row[f"{phase}_cross_entropy"] = entry[phase].get("cross_entropy", entry[phase]["loss"])
            row[f"{phase}_soft_dice"] = entry[phase].get("soft_dice", 0.0)
            row[f"{phase}_seconds"] = entry[phase]["seconds"]
            for taxonomy in ("el29", "el10"):
                for metric in ("macro_iou", "macro_dice", "defect_macro_iou", "defect_macro_dice"):
                    row[f"{phase}_{taxonomy}_{metric}"] = entry[phase][taxonomy][metric]
        rows.append(row)
    write_csv(run_dir / "history.csv", rows)
    plot_history(run_dir, history)


def save_checkpoint(path, model, optimizer, epoch, best_loss, best_epoch, history, config):
    checkpoint = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
        "best_val_loss": best_loss, "best_epoch": best_epoch, "history": history, "config": config,
        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def snapshot_sources(run_dir):
    destination = run_dir / "source"
    for folder in ("datasets", "training", "evaluation", "models/baselines", "models/custom"):
        for source in (ROOT / folder).rglob("*.py"):
            target = destination / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    source = ROOT.parent / "CAFUNet/ca_funet.py"
    if source.is_file():
        (destination / "CAFUNet").mkdir(exist_ok=True)
        shutil.copy2(source, destination / "CAFUNet/ca_funet.py")


def train(args):
    device = choose_device(args.device)
    resumed = None
    if args.resume:
        checkpoint_path = Path(args.resume).resolve()
        resumed = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = resumed["config"]
        run_dir = checkpoint_path.parent.parent
        if checkpoint_path.name != "last.pt":
            raise ValueError("Resume from checkpoints/last.pt, not best.pt")
        if args.epochs is not None:
            if args.epochs < config["epochs"]:
                raise ValueError("Cannot reduce the recorded training budget on resume")
            config["epochs"] = args.epochs
        if args.data_root:
            config["data_root"] = str(Path(args.data_root).resolve())
        manifest = json.loads((run_dir / "splits.json").read_text())
    else:
        if args.model is None:
            raise ValueError("--model is required for a new training run")
        root = Path(args.data_root or ROOT / "data/BenchmarkELimages/dataset_20221008").resolve()
        manifest = prepare_manifest(root, args.split, args.split_seed)
        run_name = args.run_name or f"{args.initialization}-{args.variant}-seed{args.seed}-{timestamp()}"
        if Path(run_name).name != run_name or run_name in (".", ".."):
            raise ValueError("--run-name must be a single directory name")
        run_dir = Path(args.output_root).resolve() / args.model / run_name
        config = {
            "model": args.model, "initialization": args.initialization, "variant": args.variant,
            "architecture_version": "solar-unet-v1" if args.model == "solar-unet" else args.model,
            "loss": args.loss,
            "loss_protocol": "ce: pixel-mean cross entropy. ce-dice: CE + equal-coefficient batch soft Dice loss over GT-present non-background classes; ignored pixels excluded; CE still penalizes absent-class false positives.",
            "seed": args.seed, "split": args.split, "split_seed": args.split_seed,
            "epochs": args.epochs or 50, "batch_size": args.batch_size or (4 if args.model == "segformer-b0" else 8),
            "learning_rate": args.learning_rate if args.learning_rate is not None else (5e-5 if args.model == "segformer-b0" else 1e-4),
            "weight_decay": 0.01, "data_root": str(root), "num_classes": NUM_CLASSES,
            "class_names": list(CLASS_NAMES), "ignore_index": 255,
            "split_sample_list_sha256": manifest["sample_list_sha256"], "checkpoint_selection": "minimum validation cross-entropy",
            "input": "PIL grayscale, float32 /255; no resizing; SegFormer ImageNet RGB normalization; pretrained ResNet grayscale (x-0.449)/0.226",
            "initialization_source": ("random" if args.initialization == "scratch" else "nvidia/segformer-b0-finetuned-ade-512-512" if args.model == "segformer-b0" else "ImageNet ResNet-34; SMP/timm default pretrained weights"),
        }
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "checkpoints").mkdir()
        write_json(run_dir / "splits.json", manifest)
        snapshot_sources(run_dir)
    if config["epochs"] <= 0 or config["batch_size"] <= 0 or config["learning_rate"] <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive")
    seed_everything(config["seed"])
    datasets = load_datasets(config["data_root"], manifest)
    audit = audit_datasets(datasets)
    if config["split"] == "module_grouped" and any(audit["module_overlap"].values()):
        raise ValueError("Module leakage detected in saved grouped split")
    if resumed:
        previous_audit = json.loads((run_dir / "dataset_audit.json").read_text())
        if previous_audit["splits"] != audit["splits"]:
            raise ValueError("Dataset content changed since the original run")
    write_json(run_dir / "dataset_audit.json", audit)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / (f"resume_environment_{timestamp()}.json" if resumed else "environment.json"), environment(device))
    for warning in audit["warnings"]:
        print(f"WARNING: {warning}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    model = build_model(config["model"], config["initialization"], config["variant"], load_pretrained=resumed is None).to(device)
    objective = SegmentationObjective(config.get("loss", "ce"))
    write_json(run_dir / "model_stats.json", {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    })
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    history, best_loss, best_epoch, start_epoch = [], math.inf, None, 1
    if resumed:
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        history, best_loss, best_epoch = resumed["history"], resumed["best_val_loss"], resumed["best_epoch"]
        start_epoch = resumed["epoch"] + 1
        torch.set_rng_state(resumed["torch_rng"])
        np.random.set_state(resumed["numpy_rng"])
        random.setstate(resumed["python_rng"])
        if device.type == "cuda" and resumed["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(resumed["cuda_rng"])
    if start_epoch > config["epochs"]:
        raise ValueError("Run already reached its epoch budget; use --epochs with a larger budget to continue")
    validation_loader = make_loader(datasets["val"], config["batch_size"], args.workers, device)
    write_json(run_dir / "status.json", {"state": "training", "next_epoch": start_epoch})
    try:
        for epoch in range(start_epoch, config["epochs"] + 1):
            training_loader = make_loader(datasets["train"], config["batch_size"], args.workers, device, True, config["seed"] + epoch)
            train_report = run_epoch(model, training_loader, device, optimizer, objective=objective)
            val_report = run_epoch(model, validation_loader, device, objective=objective)
            history.append({"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train": train_report, "val": val_report})
            improved = val_report["cross_entropy"] < best_loss
            if improved:
                best_loss, best_epoch = val_report["cross_entropy"], epoch
            save_checkpoint(run_dir / "checkpoints/last.pt", model, optimizer, epoch, best_loss, best_epoch, history, config)
            if improved:
                shutil.copy2(run_dir / "checkpoints/last.pt", run_dir / "checkpoints/best.pt")
                write_json(run_dir / "best_validation.json", history[-1])
            save_history(run_dir, history)
            write_json(run_dir / "status.json", {"state": "training", "completed_epoch": epoch, "best_epoch": best_epoch})
            print(f"epoch={epoch}/{config['epochs']} train_loss={train_report['loss']:.5f} val_loss={val_report['loss']:.5f} val_ce={val_report['cross_entropy']:.5f} val_mIoU={val_report['el29']['macro_iou']:.5f} best_epoch={best_epoch}", flush=True)
    except BaseException as error:
        write_json(run_dir / "status.json", {"state": "interrupted", "error": str(error), "last_completed_epoch": history[-1]["epoch"] if history else None})
        raise
    write_json(run_dir / "status.json", {"state": "completed", "completed_epoch": config["epochs"], "best_epoch": best_epoch})
    print(f"Training complete. Evaluate explicitly: python -m training.run evaluate --run-dir '{run_dir}' --eval-split test", flush=True)
    return run_dir


def evaluate(args):
    run_dir = Path(args.run_dir).resolve()
    config = json.loads((run_dir / "config.json").read_text())
    manifest = json.loads((run_dir / "splits.json").read_text())
    root = args.data_root or config["data_root"]
    datasets = load_datasets(root, manifest)
    audit = audit_datasets(datasets)
    previous_audit = json.loads((run_dir / "dataset_audit.json").read_text())
    if previous_audit["splits"] != audit["splits"]:
        raise ValueError("Dataset content differs from the training run")
    device = choose_device(args.device)
    model = build_model(config["model"], config["initialization"], config["variant"], load_pretrained=False).to(device)
    checkpoint = torch.load(run_dir / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    directory = run_dir / "evaluation" / f"{args.eval_split}-{timestamp()}"
    directory.mkdir(parents=True, exist_ok=False)
    loader = make_loader(datasets[args.eval_split], args.batch_size or config["batch_size"], args.workers, device)
    report = run_epoch(model, loader, device, output_dir=directory, figures=args.figures, objective=SegmentationObjective(config.get("loss", "ce")))
    report.update({"split": args.eval_split, "checkpoint": "checkpoints/best.pt", "epoch": checkpoint["epoch"], "sample_list_sha256": manifest["sample_list_sha256"], "dataset_coverage": audit["splits"][args.eval_split]})
    save_evaluation(directory, report)
    write_json(directory / "environment.json", environment(device))
    write_json(directory / "palette.json", {entry.label: {"name": entry.name, "rgb": list(entry.rgb)} for entry in EL_CLASSES})
    print(f"{args.eval_split}: EL-29 mIoU={report['el29']['macro_iou']} Dice={report['el29']['macro_dice']}\nSaved report: {directory}", flush=True)
    return directory


def cleanup_legacy(output_root):
    output_root = Path(output_root).resolve()
    legacy_root = output_root.parent
    candidates = [legacy_root / name for name in ("U-Net", "U-Net++", "DeepLabV3+", "SegFormer-B0", "class_stats.npz")]
    candidates = [path for path in candidates if path.exists()]
    if not candidates:
        print("No legacy artifacts found; new runs and notebook logs are untouched.")
        return
    destination = legacy_root / "legacy" / timestamp()
    destination.mkdir(parents=True)
    for path in candidates:
        shutil.move(str(path), destination / path.name)
    write_json(destination / "archive.json", {"reason": "Legacy runs predate shared evaluation; retain weights and evidence, do not compare as protocol-v1 runs", "artifacts": [path.name for path in candidates]})
    print(f"Archived old artifacts without deleting checkpoints: {destination}")


def smoke(args):
    if args.model is None:
        raise ValueError("smoke requires --model")
    seed_everything(args.seed)
    device = choose_device(args.device)
    root = args.data_root or ROOT / "data/BenchmarkELimages/dataset_20221008"
    manifest = prepare_manifest(root, args.split, args.split_seed)
    datasets = load_datasets(root, manifest)
    model = build_model(args.model, args.initialization, args.variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate or 1e-4)
    report = {"purpose": "One training batch and one validation batch only; NOT benchmark results", "model": args.model, "initialization": args.initialization, "variant": args.variant, "loss": args.loss, "parameters": sum(parameter.numel() for parameter in model.parameters())}
    for phase in ("train", "val"):
        loader = make_loader(datasets[phase], args.batch_size or 2, args.workers, device)
        report[phase] = run_epoch(model, [next(iter(loader))], device, optimizer if phase == "train" else None, objective=SegmentationObjective(args.loss))
    directory = Path(args.output_root) / "smoke" / args.model / timestamp()
    write_json(directory / "smoke.json", report)
    write_json(directory / "environment.json", environment(device))
    print(f"SMOKE PASSED: {args.model}, {args.initialization}, {args.variant}; {directory}")
    return directory


def main(argv=None, default_model=None):
    parser = argparse.ArgumentParser(description="Reproducible EL-29 training and explicit evaluation")
    parser.add_argument("action", choices=("train", "evaluate", "audit", "smoke", "cleanup"), nargs="?", default="train")
    parser.add_argument("--model", choices=MODEL_NAMES, default=default_model)
    parser.add_argument("--data-root", "--root", dest="data_root")
    parser.add_argument("--output-root", default=str(ROOT / "results/runs"))
    parser.add_argument("--run-name")
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", help="Trusted local checkpoints/last.pt only")
    parser.add_argument("--init", dest="initialization", choices=("scratch", "pretrained"), default="scratch")
    parser.add_argument("--variant", choices=("plain", "detail", "context", "fixed", "dynamic"), default="dynamic")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--loss", choices=("ce", "ce-dice"), default="ce")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split", choices=("module_grouped", "official"), default="module_grouped")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu", "mps"))
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")
    parser.add_argument("--figures", type=int, default=12, help="Number of qualitative image panels; all indexed predictions are saved")
    args = parser.parse_args(argv)
    if args.workers < 0 or args.figures < 0 or (args.batch_size is not None and args.batch_size <= 0) or (args.epochs is not None and args.epochs <= 0):
        parser.error("Workers/figures must be nonnegative; batch size and epochs must be positive")
    if args.action in ("train", "smoke") and not args.resume and args.model != "solar-unet" and args.variant in ("detail", "context"):
        parser.error("detail/context variants require --model solar-unet")
    if args.action == "cleanup":
        return cleanup_legacy(args.output_root)
    if args.action == "smoke":
        return smoke(args)
    if args.action == "audit":
        root = args.data_root or ROOT / "data/BenchmarkELimages/dataset_20221008"
        manifest = prepare_manifest(root, args.split, args.split_seed)
        report = audit_datasets(load_datasets(root, manifest))
        directory = Path(args.output_root) / "audits" / timestamp()
        write_json(directory / "splits.json", manifest)
        write_json(directory / "dataset_audit.json", report)
        print(json.dumps(report, indent=2))
        print(f"Audit saved to {directory}")
        return directory
    if args.action == "evaluate":
        if not args.run_dir:
            parser.error("evaluate requires --run-dir")
        return evaluate(args)
    return train(args)


if __name__ == "__main__":
    main()