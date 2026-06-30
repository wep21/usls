from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn


HF_BASE = "https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft"

VARIANTS = {
    "ev-m": {
        "checkpoint": "efficientsam3_efficientvit.pt",
        "backbone_type": "efficientvit",
        "model_name": "b1",
    },
    "rv-m": {
        "checkpoint": "efficientsam3_repvit.pt",
        "backbone_type": "repvit",
        "model_name": "m1.1",
    },
    "tv-m": {
        "checkpoint": "efficientsam3_tinyvit.pt",
        "backbone_type": "tinyvit",
        "model_name": "11m",
    },
}


MODULE_STEMS = {
    "vision": "vision-encoder",
    "text": "text-encoder",
    "decoder": "decoder",
}


def add_upstream_repo(repo_root: str | None) -> None:
    if repo_root is None:
        return
    root = Path(repo_root).expanduser().resolve()
    package_root = root / "sam3" if (root / "sam3" / "sam3").is_dir() else root
    sys.path.insert(0, str(package_root))


def cached_checkpoint(variant: str, explicit: str | None) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    filename = VARIANTS[variant]["checkpoint"]
    out = (
        Path.home()
        / ".cache"
        / "usls"
        / "Simon7108528"
        / "EfficientSAM3"
        / "efficientsam3_ft"
        / filename
    )
    if out.is_file():
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF_BASE}/{filename}?download=true"
    print(f"Downloading {url} -> {out}")
    urllib.request.urlretrieve(url, out)
    return out


class EfficientSam3VisionEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone

    def forward(self, images):
        out = self.backbone.forward_image(images)
        fpn = out["backbone_fpn"]
        pos = out["vision_pos_enc"]
        return fpn[0], fpn[1], fpn[2], pos[2]


class EfficientSam3TextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text = model.backbone.language_backbone

    def forward(self, input_ids, attention_mask):
        del attention_mask
        input_embeds = self.text.encoder.forward_embedding(input_ids)
        text_features = self.text.encoder(
            input_embeds,
            return_all_tokens=True,
            input_is_embeddings=True,
        )
        text_features = self.text.projector(text_features)
        text_mask = input_ids == 0
        return text_features, text_mask


class EfficientSam3Decoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.geometry_encoder = model.geometry_encoder
        self.transformer = model.transformer
        self.segmentation_head = model.segmentation_head
        self.dot_product_scoring = model.dot_prod_scoring

    def forward(
        self,
        fpn_feat_0,
        fpn_feat_1,
        fpn_feat_2,
        fpn_pos_2,
        text_features,
        text_mask,
        input_boxes,
        input_boxes_labels,
    ):
        batch = fpn_feat_2.shape[0]
        feat_h, feat_w = fpn_feat_2.shape[-2:]

        img_feat = fpn_feat_2.flatten(2).permute(2, 0, 1)
        img_pos = fpn_pos_2.flatten(2).permute(2, 0, 1)

        box_padding = input_boxes_labels == -10
        box_labels = torch.where(
            box_padding,
            torch.zeros_like(input_boxes_labels),
            input_boxes_labels,
        )

        from sam3.model.geometry_encoders import Prompt

        geo_prompt = Prompt(
            box_embeddings=input_boxes.transpose(0, 1),
            box_mask=box_padding,
            box_labels=box_labels.transpose(0, 1),
        )
        geometry_features, geometry_mask = self.geometry_encoder(
            geo_prompt=geo_prompt,
            img_feats=[img_feat],
            img_sizes=[(feat_h, feat_w)],
            img_pos_embeds=[img_pos],
        )

        prompt = torch.cat([text_features.transpose(0, 1), geometry_features], dim=0)
        prompt_mask = torch.cat([text_mask, geometry_mask], dim=1)
        prompt_pos = torch.zeros_like(prompt)

        memory = self.transformer.encoder(
            src=[img_feat],
            src_key_padding_mask=None,
            src_pos=[img_pos],
            prompt=prompt,
            prompt_pos=prompt_pos,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=[(feat_h, feat_w)],
        )

        query_embed = self.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, batch, 1)
        hs, reference_boxes, presence_logits, _ = self.transformer.decoder(
            tgt=tgt,
            memory=memory["memory"],
            memory_key_padding_mask=memory["padding_mask"],
            pos=memory["pos_embed"],
            reference_boxes=None,
            level_start_index=memory["level_start_index"],
            spatial_shapes=memory["spatial_shapes"],
            valid_ratios=memory["valid_ratios"],
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            apply_dac=False,
        )

        hs = hs.transpose(1, 2)
        reference_boxes = reference_boxes.transpose(1, 2)
        if presence_logits is None:
            presence = torch.zeros(batch, 1, device=fpn_feat_2.device)
        else:
            presence = presence_logits.transpose(1, 2)[-1]

        pred_logits = self.dot_product_scoring(
            hs,
            prompt,
            prompt_mask,
        )[-1].squeeze(-1)

        box_offsets = self.transformer.decoder.bbox_embed(hs)
        pred_boxes = self._box_cxcywh_to_xyxy(
            (self._inverse_sigmoid(reference_boxes) + box_offsets).sigmoid()
        )[-1]

        masks = self.segmentation_head(
            backbone_feats=[fpn_feat_0, fpn_feat_1, fpn_feat_2],
            obj_queries=hs,
            image_ids=torch.arange(batch, device=fpn_feat_2.device),
            encoder_hidden_states=memory["memory"],
            prompt=prompt,
            prompt_mask=prompt_mask,
        )["pred_masks"]

        return masks, pred_boxes, pred_logits, presence

    @staticmethod
    def _inverse_sigmoid(x, eps=1e-3):
        x = x.clamp(min=0, max=1)
        return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))

    @staticmethod
    def _box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(-1)
        return torch.stack(
            (x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h),
            dim=-1,
        )


def build_model(args):
    from sam3.model_builder import build_efficientsam3_image_model

    spec = VARIANTS[args.variant]
    checkpoint = cached_checkpoint(args.variant, args.checkpoint)
    model = build_efficientsam3_image_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        device=args.device,
        eval_mode=True,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        backbone_type=spec["backbone_type"],
        model_name=spec["model_name"],
        text_encoder_type="MobileCLIP-S0",
        text_encoder_context_length=args.text_context_length,
    )
    return model.eval()


def module_path(output_dir: Path, module: str, args) -> Path:
    stem = MODULE_STEMS[module]
    if args.file_prefix:
        stem = f"{args.file_prefix}-{stem}"
    if args.file_dtype_suffix:
        stem = f"{stem}-{args.file_dtype_suffix}"
    return output_dir / f"{stem}.onnx"


def maybe_convert_fp16(path: Path, args) -> None:
    if args.precision != "fp16":
        return

    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(path))
    model = float16.convert_float_to_float16(model)
    onnx.save(model, str(path))


def export_vision(model, output_dir: Path, args) -> None:
    wrapper = EfficientSam3VisionEncoder(model).to(args.device).eval()
    out = module_path(output_dir, "vision", args)
    torch.onnx.export(
        wrapper,
        (torch.randn(1, 3, args.image_height, args.image_width, device=args.device),),
        str(out),
        input_names=["images"],
        output_names=["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={
            "images": {0: "batch"},
            "fpn_feat_0": {0: "batch"},
            "fpn_feat_1": {0: "batch"},
            "fpn_feat_2": {0: "batch"},
            "fpn_pos_2": {0: "batch"},
        },
    )
    maybe_convert_fp16(out, args)


def export_text(model, output_dir: Path, args) -> None:
    wrapper = EfficientSam3TextEncoder(model).to(args.device).eval()
    seq = args.text_context_length
    out = module_path(output_dir, "text", args)
    torch.onnx.export(
        wrapper,
        (
            torch.randint(0, 49408, (1, seq), device=args.device, dtype=torch.long),
            torch.ones(1, seq, dtype=torch.long, device=args.device),
        ),
        str(out),
        input_names=["input_ids", "attention_mask"],
        output_names=["text_features", "text_mask"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "text_features": {0: "batch"},
            "text_mask": {0: "batch"},
        },
    )
    maybe_convert_fp16(out, args)


def export_decoder(model, output_dir: Path, args) -> None:
    wrapper = EfficientSam3Decoder(model).to(args.device).eval()
    patch_h = args.image_height // 14
    patch_w = args.image_width // 14
    seq = args.text_context_length
    out = module_path(output_dir, "decoder", args)
    torch.onnx.export(
        wrapper,
        (
            torch.randn(1, 256, patch_h * 4, patch_w * 4, device=args.device),
            torch.randn(1, 256, patch_h * 2, patch_w * 2, device=args.device),
            torch.randn(1, 256, patch_h, patch_w, device=args.device),
            torch.randn(1, 256, patch_h, patch_w, device=args.device),
            torch.randn(1, seq, 256, device=args.device),
            torch.zeros(1, seq, dtype=torch.bool, device=args.device),
            torch.rand(1, 5, 4, device=args.device),
            torch.ones(1, 5, dtype=torch.long, device=args.device),
        ),
        str(out),
        input_names=[
            "fpn_feat_0",
            "fpn_feat_1",
            "fpn_feat_2",
            "fpn_pos_2",
            "text_features",
            "text_mask",
            "input_boxes",
            "input_boxes_labels",
        ],
        output_names=["pred_masks", "pred_boxes", "pred_logits", "presence_logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={
            "fpn_feat_0": {0: "batch"},
            "fpn_feat_1": {0: "batch"},
            "fpn_feat_2": {0: "batch"},
            "fpn_pos_2": {0: "batch"},
            "text_features": {0: "batch"},
            "text_mask": {0: "batch"},
            "input_boxes": {0: "batch", 1: "num_boxes"},
            "input_boxes_labels": {0: "batch", 1: "num_boxes"},
            "pred_masks": {0: "batch"},
            "pred_boxes": {0: "batch"},
            "pred_logits": {0: "batch"},
            "presence_logits": {0: "batch"},
        },
    )
    maybe_convert_fp16(out, args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export EfficientSAM3 image models to ONNX")
    parser.add_argument("--variant", choices=VARIANTS.keys(), default="tv-m")
    parser.add_argument("--checkpoint", help="Path to a downloaded EfficientSAM3 .pt checkpoint")
    parser.add_argument(
        "--repo-root",
        help="Path to a cloned SimonZeng7108/efficientsam3 repo or its sam3 package directory",
    )
    parser.add_argument(
        "--module",
        choices=["vision", "text", "decoder"],
        default=None,
    )
    parser.add_argument("--all", action="store_true", help="Export all modules")
    parser.add_argument("--output-dir", default="onnx-efficient-sam3")
    parser.add_argument(
        "--file-prefix",
        help="Prefix output names for release assets, e.g. efficient-sam3-tv-m",
    )
    parser.add_argument(
        "--file-dtype-suffix",
        help="Optional dtype suffix before .onnx, e.g. fp16",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16"],
        default="fp32",
        help="Post-convert exported ONNX float tensors to fp16 when set to fp16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-height", type=int, default=1008)
    parser.add_argument("--image-width", type=int, default=1008)
    parser.add_argument("--text-context-length", type=int, default=16)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    if not args.module and not args.all:
        parser.error("Please specify --module or --all")
    if args.image_height % 14 != 0 or args.image_width % 14 != 0:
        raise ValueError("image height and width must be multiples of 14")
    if args.file_dtype_suffix is None and args.precision != "fp32":
        args.file_dtype_suffix = args.precision

    add_upstream_repo(args.repo_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args)
    modules = ["vision", "text", "decoder"] if args.all else [args.module]
    with torch.no_grad():
        for module in modules:
            if module == "vision":
                export_vision(model, output_dir, args)
            elif module == "text":
                export_text(model, output_dir, args)
            elif module == "decoder":
                export_decoder(model, output_dir, args)


if __name__ == "__main__":
    main()
