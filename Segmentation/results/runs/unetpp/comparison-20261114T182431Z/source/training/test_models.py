import importlib.util
import gc
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import torch

from models.baselines.unet import UNet
from models.custom.solar_unet import ClassSpatialFusion, ELEdgeFeatures, SOLAR_VARIANTS, SolarUNet
from training.models import MODEL_NAMES, build_model


class StubEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_info = types.SimpleNamespace(channels=lambda: [4, 8, 16, 32, 64])

    def forward(self, images):
        return [
            torch.nn.functional.avg_pool2d(images, stride).repeat(1, channels, 1, 1)
            for channels, stride in zip(self.feature_info.channels(), (2, 4, 8, 16, 32))
        ]


class ModelContractTests(unittest.TestCase):
    def test_el_edges_preserve_flat_fields_and_respond_to_thin_lines(self):
        extractor = ELEdgeFeatures()
        flat = torch.full((1, 1, 32, 32), 0.5)
        edges = extractor(flat)
        torch.testing.assert_close(edges[:, :1], flat)
        torch.testing.assert_close(edges[:, 1:], torch.zeros(1, 2, 32, 32), atol=1e-6, rtol=0)
        cracked = flat.clone()
        cracked[:, :, :, 16] = 0
        self.assertGreater(float(extractor(cracked)[:, 1:].abs().sum()), 0)

    def test_fusion_has_independent_class_and_pixel_weights(self):
        fusion = ClassSpatialFusion(8, 3)
        semantic = torch.randn(2, 8, 9, 11)
        detail = torch.randn_like(semantic)
        weights = fusion.weights(semantic, detail)
        self.assertEqual(tuple(weights.shape), (2, 3, 9, 11))
        self.assertTrue(bool(((weights > 0) & (weights < 1)).all()))
        self.assertGreater(float(weights.var(dim=1).mean()), 0)
        self.assertGreater(float(weights.flatten(2).var(dim=2).mean()), 0)

    @unittest.skipUnless(importlib.util.find_spec("timm"), "Install timm to test SolarUNet")
    def test_solar_factory_reload_has_no_pretrained_download(self):
        with patch("timm.create_model", return_value=StubEncoder()) as constructor:
            model = build_model("solar-unet", initialization="pretrained", load_pretrained=False)
        self.assertFalse(constructor.call_args.kwargs["pretrained"])
        self.assertTrue(model.normalize_encoder)
        self.assertEqual(model.variant, "dynamic")

    def test_solar_only_variants_are_not_silent_noops(self):
        with self.assertRaisesRegex(ValueError, "specific to solar-unet"):
            build_model("cafunet", variant="context")

    @unittest.skipUnless(importlib.util.find_spec("timm"), "Install timm to test SolarUNet")
    def test_solar_variants_forward_backward_and_odd_shapes(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            for variant in SOLAR_VARIANTS:
                with self.subTest(variant=variant):
                    model = SolarUNet(variant=variant)
                    logits = model(torch.rand(2, 1, 65, 97))
                    self.assertEqual(tuple(logits.shape), (2, 29, 65, 97))
                    loss = torch.nn.functional.cross_entropy(logits, torch.zeros(2, 65, 97, dtype=torch.long))
                    self.assertTrue(bool(torch.isfinite(loss)))
                    loss.backward()
                    if model.use_detail:
                        self.assertGreater(float(model.detail_stem.layers[0].weight.grad.abs().sum()), 0)
                    if variant == "dynamic":
                        self.assertGreater(float(model.fusion.gate[-1].weight.grad.abs().sum()), 0)
                    if variant in ("context", "fixed", "dynamic"):
                        self.assertGreater(float(model.context.row.weight.grad.abs().sum()), 0)
                        self.assertGreater(float(model.context.column.weight.grad.abs().sum()), 0)
                    del model, logits, loss
                    gc.collect()
        finally:
            torch.set_num_threads(previous_threads)

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) for name in ("segmentation_models_pytorch", "timm", "transformers")),
        "Install the model requirements to run real-library contracts",
    )
    def test_real_scratch_models_forward_backward(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            for name in MODEL_NAMES:
                with self.subTest(model=name):
                    model = build_model(name, "scratch")
                    logits = model(torch.rand(2, 1, 64, 64))
                    self.assertEqual(tuple(logits.shape), (2, 29, 64, 64))
                    loss = torch.nn.functional.cross_entropy(logits, torch.zeros(2, 64, 64, dtype=torch.long))
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
                    del model, logits, loss
                    gc.collect()
        finally:
            torch.set_num_threads(previous_threads)

    def test_original_unet_forward_backward(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            model = UNet()
            logits = model(torch.rand(2, 1, 32, 32))
            self.assertEqual(tuple(logits.shape), (2, 29, 32, 32))
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(2, 32, 32, dtype=torch.long))
            loss.backward()
            self.assertIsNotNone(model.outc.weight.grad)
        finally:
            torch.set_num_threads(previous_threads)

    def test_cafunet_contract_with_stub_encoder(self):
        stub = types.ModuleType("timm")
        stub.create_model = lambda *args, **kwargs: StubEncoder()
        source = Path(__file__).resolve().parents[2] / "CAFUNet/ca_funet.py"
        specification = importlib.util.spec_from_file_location("cafunet_contract", source)
        module = importlib.util.module_from_spec(specification)
        with patch.dict("sys.modules", {"timm": stub}):
            specification.loader.exec_module(module)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            for dual, dynamic in ((False, False), (True, False), (True, True)):
                model = module.CAFUNet(
                    classes=29, pretrained=False, decoder_channels=(32, 16, 8, 4, 2),
                    use_dual_path=dual, use_dynamic_weights=dynamic,
                )
                logits = model(torch.rand(2, 1, 64, 64))
                self.assertEqual(tuple(logits.shape), (2, 29, 64, 64))
                logits.square().mean().backward()
                if dynamic:
                    self.assertIsNotNone(model.blocks[0].fusion_skip.fusion_gate.fc2.weight.grad)
            with self.assertRaisesRegex(ValueError, "requires dual paths"):
                module.DualPathCAFusionSkip(4, 4, False, True)
        finally:
            torch.set_num_threads(previous_threads)

    def test_smp_outputs_are_tensors(self):
        stub = types.ModuleType("segmentation_models_pytorch")
        constructor = unittest.mock.Mock(side_effect=lambda **kwargs: torch.nn.Conv2d(1, kwargs["classes"], 1))
        stub.Unet = stub.UnetPlusPlus = stub.DeepLabV3Plus = constructor
        with patch.dict("sys.modules", {"segmentation_models_pytorch": stub}):
            for name in ("unet-resnet34", "unetpp", "deeplabv3plus"):
                model = build_model(name)
                self.assertEqual(tuple(model(torch.zeros(1, 1, 32, 32)).shape), (1, 29, 32, 32))
        self.assertIsNone(constructor.call_args.kwargs["encoder_weights"])


if __name__ == "__main__":
    unittest.main()