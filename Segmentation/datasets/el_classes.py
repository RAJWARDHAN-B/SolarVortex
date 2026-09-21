"""Class taxonomy for the BenchmarkELimages dataset_20221008 (29 classes, labels 0-28).

Ground-truth masks in el_masks_{train,val,test} are single-channel PNGs whose pixel
values ARE the class label. No palette or RGB lookup is required.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

NUM_CLASSES = 29
IGNORE_INDEX = 255


@dataclass(frozen=True)
class ELClass:
    label: int
    name: str
    kind: str  # "feature" (intrinsic cell structure) or "defect" (extrinsic damage)
    rgb: tuple[int, int, int]


# Mirrors ListOfClassesAndColorCodes_20221008.csv, with names normalised to snake_case.
EL_CLASSES: tuple[ELClass, ...] = (
    ELClass(0, "background", "feature", (0, 0, 0)),
    ELClass(1, "sp_multi", "feature", (128, 128, 128)),
    ELClass(2, "sp_mono", "feature", (80, 80, 80)),
    ELClass(3, "sp_dogbone", "feature", (0, 0, 255)),
    ELClass(4, "ribbons", "feature", (0, 255, 0)),
    ELClass(5, "border", "feature", (100, 50, 50)),
    ELClass(6, "text", "feature", (225, 0, 100)),
    ELClass(7, "padding", "feature", (128, 128, 0)),
    ELClass(8, "clamp", "feature", (255, 215, 0)),
    ELClass(9, "busbars", "feature", (50, 50, 255)),
    ELClass(10, "crack_rbn_edge", "defect", (0, 255, 255)),
    ELClass(11, "inactive", "defect", (255, 0, 0)),
    ELClass(12, "rings", "defect", (255, 0, 255)),
    ELClass(13, "material", "defect", (255, 255, 0)),
    ELClass(14, "crack", "defect", (255, 255, 255)),
    ELClass(15, "gridline", "defect", (255, 165, 0)),
    ELClass(16, "splice", "defect", (75, 0, 130)),
    ELClass(17, "dead_cell", "defect", (32, 32, 32)),
    ELClass(18, "corrosion_rbn", "defect", (0, 150, 0)),
    ELClass(19, "belt_mark", "defect", (218, 165, 32)),
    ELClass(20, "edge_dark", "defect", (184, 134, 11)),
    ELClass(21, "frame_edge", "feature", (127, 255, 215)),
    ELClass(22, "jbox", "feature", (45, 45, 255)),
    ELClass(23, "meas_artifact", "defect", (50, 50, 50)),
    ELClass(24, "sp_mono_halfcut", "feature", (100, 100, 100)),
    ELClass(25, "scuff", "defect", (200, 200, 0)),
    ELClass(26, "corrosion_cell", "defect", (0, 100, 0)),
    ELClass(27, "brightening", "defect", (192, 192, 192)),
    ELClass(28, "star", "defect", (200, 0, 200)),
)

CLASS_NAMES: tuple[str, ...] = tuple(c.name for c in EL_CLASSES)
FEATURE_LABELS: tuple[int, ...] = tuple(c.label for c in EL_CLASSES if c.kind == "feature")
DEFECT_LABELS: tuple[int, ...] = tuple(c.label for c in EL_CLASSES if c.kind == "defect")


# ---------------------------------------------------------------------------
# EL-10: reduced taxonomy
# ---------------------------------------------------------------------------
# Six of the 29 classes have zero pixels in the official test split (clamp,
# crack_rbn_edge, dead_cell, belt_mark, meas_artifact, sp_mono_halfcut), so a
# 29-class macro-mIoU averages over undefined terms. EL-10 merges classes that
# share a defect mechanism and appearance, keeping every group non-empty in all
# three splits. meas_artifact is a camera artefact rather than a cell property
# and is mapped to IGNORE_INDEX so it contributes to neither loss nor metrics.
EL10_NAMES: tuple[str, ...] = (
    "background",       # 0
    "cell_substrate",   # 1  wafer body, all mono/multi/dogbone/halfcut variants
    "interconnect",     # 2  ribbons + busbars
    "module_furniture", # 3  border, text, padding, clamp, frame_edge, jbox
    "crack",            # 4  crack + crack at ribbon edge
    "inactive",         # 5  inactive area + fully dead cell
    "gridline",         # 6  broken/lost gridline fingers
    "corrosion",        # 7  ribbon + cell corrosion
    "surface_anomaly",  # 8  material, scuff, belt_mark, brightening, rings, star, splice
    "edge_dark",        # 9  darkened cell edge
)

NUM_EL10_CLASSES = len(EL10_NAMES)

EL10_GROUPS: dict[str, tuple[int, ...]] = {
    "background": (0,),
    "cell_substrate": (1, 2, 3, 24),
    "interconnect": (4, 9),
    "module_furniture": (5, 6, 7, 8, 21, 22),
    "crack": (10, 14),
    "inactive": (11, 17),
    "gridline": (15,),
    "corrosion": (18, 26),
    "surface_anomaly": (12, 13, 16, 19, 25, 27, 28),
    "edge_dark": (20,),
}

EL10_IGNORED: tuple[int, ...] = (23,)  # meas_artifact

EL10_DEFECT_NAMES: tuple[str, ...] = (
    "crack",
    "inactive",
    "gridline",
    "corrosion",
    "surface_anomaly",
    "edge_dark",
)


def build_el10_lut() -> list[int]:
    """Return a 29-entry lookup table mapping a fine label to its EL-10 label."""
    lut = [IGNORE_INDEX] * NUM_CLASSES
    for new_label, name in enumerate(EL10_NAMES):
        for fine in EL10_GROUPS[name]:
            lut[fine] = new_label
    for fine in EL10_IGNORED:
        lut[fine] = IGNORE_INDEX
    unassigned = [i for i, v in enumerate(lut) if v == IGNORE_INDEX and i not in EL10_IGNORED]
    if unassigned:
        raise ValueError(f"labels not covered by EL10_GROUPS: {unassigned}")
    return lut


EL10_LUT: tuple[int, ...] = tuple(build_el10_lut())


def load_class_csv(csv_path: str | Path) -> list[ELClass]:
    """Re-read the shipped CSV so a dataset version bump is caught instead of assumed."""
    out: list[ELClass] = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(
                ELClass(
                    label=int(row["Label"]),
                    name=row["Desc"].strip().replace(" ", "_").replace("-", "_"),
                    kind=row["Class"].strip(),
                    rgb=(int(row["Red"]), int(row["Green"]), int(row["Blue"])),
                )
            )
    return sorted(out, key=lambda c: c.label)


def verify_against_csv(csv_path: str | Path) -> None:
    """Raise if the shipped CSV disagrees with the hardcoded taxonomy."""
    shipped = load_class_csv(csv_path)
    if len(shipped) != NUM_CLASSES:
        raise ValueError(f"CSV has {len(shipped)} classes, expected {NUM_CLASSES}")
    for got, want in zip(shipped, EL_CLASSES):
        if got.label != want.label or got.kind != want.kind or got.rgb != want.rgb:
            raise ValueError(f"CSV/taxonomy mismatch at label {got.label}: {got} != {want}")
