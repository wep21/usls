# EfficientSAM3 ONNX Export

Export EfficientSAM3 image checkpoints from `SimonZeng7108/efficientsam3` to the three ONNX files used by the Rust `EfficientSAM3` preset.

By default, the Rust presets fetch ONNX weights from the `wep21/assets` GitHub release tag `efficient-sam3`.

## Reference

- **EfficientSAM3 release:** https://github.com/SimonZeng7108/efficientsam3/releases/tag/v0.4.0-efficientsam3ft-20260611
- **Hugging Face checkpoints:** https://huggingface.co/Simon7108528/EfficientSAM3/tree/main/efficientsam3_ft
- **ONNX assets:** https://github.com/wep21/assets/releases/tag/efficient-sam3

## Export

EfficientSAM3 release checkpoints are published as PyTorch `.pt` files, not ONNX files. Clone the upstream package first, then export one of the supported variants:

| Variant | Checkpoint | Vision encoder | Text encoder |
|---------|------------|----------------|--------------|
| `ev-m` | `efficientsam3_efficientvit.pt` | EfficientViT-B1 | MobileCLIP-S0 |
| `rv-m` | `efficientsam3_repvit.pt` | RepViT-M1.1 | MobileCLIP-S0 |
| `tv-m` | `efficientsam3_tinyvit.pt` | TinyViT-11M | MobileCLIP-S0 |

```bash
git clone https://github.com/SimonZeng7108/efficientsam3 /tmp/efficientsam3

cd scripts/efficient-sam3
uv run export.py --all \
  --variant tv-m \
  --repo-root /tmp/efficientsam3 \
  --output-dir onnx-efficient-sam3-tv-m \
  --file-prefix efficient-sam3-tv-m \
  --device cuda \
  --image-height 1008 --image-width 1008
```

`uv` installs the exporter dependencies from this directory's `pyproject.toml`, and `--repo-root` points the script at the cloned upstream EfficientSAM3 package. The script downloads the selected checkpoint from `Simon7108528/EfficientSAM3/efficientsam3_ft` unless `--checkpoint` is provided.

Upload the generated files to `wep21/assets` release tag `efficient-sam3`:

- `efficient-sam3-ev-m-vision-encoder.onnx`
- `efficient-sam3-ev-m-text-encoder.onnx`
- `efficient-sam3-ev-m-decoder.onnx`
- `efficient-sam3-rv-m-vision-encoder.onnx`
- `efficient-sam3-rv-m-text-encoder.onnx`
- `efficient-sam3-rv-m-decoder.onnx`
- `efficient-sam3-tv-m-vision-encoder.onnx`
- `efficient-sam3-tv-m-text-encoder.onnx`
- `efficient-sam3-tv-m-decoder.onnx`

For fp16 assets, export with `--precision fp16`. The output names get the `-fp16` suffix that the Rust dtype resolver already probes when `--*-dtype f16` is used:

```bash
uv run export.py --all \
  --variant tv-m \
  --repo-root /tmp/efficientsam3 \
  --output-dir onnx-efficient-sam3-tv-m-fp16 \
  --file-prefix efficient-sam3-tv-m \
  --precision fp16 \
  --device cuda \
  --image-height 1008 --image-width 1008
```

This produces `efficient-sam3-tv-m-vision-encoder-fp16.onnx`, `efficient-sam3-tv-m-text-encoder-fp16.onnx`, and `efficient-sam3-tv-m-decoder-fp16.onnx`.

If the release does not exist yet, create it once:

```bash
gh release create efficient-sam3 \
  --repo wep21/assets \
  --title "EfficientSAM3 ONNX" \
  --notes "EfficientSAM3 ONNX weights for usls"
```

Upload generated weights:

```bash
gh release upload efficient-sam3 \
  onnx-efficient-sam3-tv-m/*.onnx \
  onnx-efficient-sam3-tv-m-fp16/*.onnx \
  --repo wep21/assets \
  --clobber
```

!!! note "Tokenizer"
    EfficientSAM3 and SAM3-LiteText use the MobileCLIP/OpenAI CLIP BPE tokenizer (`bpe_simple_vocab_16e6.txt.gz` upstream), not the SAM3 Hugging Face tokenizer. The Rust preset uses the existing `clip/tokenizer.json` assets with a fixed context length of 16 and zero padding to match the released MobileCLIP-S0 models.
