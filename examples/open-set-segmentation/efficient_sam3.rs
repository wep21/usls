use anyhow::Result;
use clap::{Args, ValueEnum};
use std::path::PathBuf;
use usls::{Config, DType, Device};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum EfficientSam3Variant {
    /// EV-M: EfficientViT-B1 + MobileCLIP-S0.
    EvM,
    /// RV-M: RepViT-M1.1 + MobileCLIP-S0.
    RvM,
    /// TV-M: TinyViT-11M + MobileCLIP-S0.
    TvM,
}

#[derive(Args, Debug)]
pub struct EfficientSam3Args {
    /// EfficientSAM3 model variant.
    #[arg(long, value_enum, default_value = "tv-m")]
    pub variant: EfficientSam3Variant,

    /// Optional local directory containing vision-encoder.onnx, text-encoder.onnx, and decoder.onnx.
    /// If omitted, weights are fetched from wep21/assets.
    #[arg(long)]
    pub model_dir: Option<PathBuf>,

    /// Visual Encoder Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "f16")]
    pub visual_encoder_dtype: DType,

    /// Visual Encoder Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub visual_encoder_device: Device,

    /// Visual encoder batch
    #[arg(long, default_value_t = 1)]
    pub visual_encoder_batch: usize,

    /// Textual Encoder Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "fp16")]
    pub textual_encoder_dtype: DType,

    /// Textual Encoder Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub textual_encoder_device: Device,

    /// Textual encoder batch
    #[arg(long, default_value_t = 1)]
    pub textual_encoder_batch: usize,

    /// Decoder Dtype: fp32, fp16, q4f16, etc.
    #[arg(long, default_value = "f16")]
    pub decoder_dtype: DType,

    /// Decoder Device: cpu, cuda:0, mps, coreml, openvino:CPU, etc.
    #[arg(long, global = true, default_value = "cpu")]
    pub decoder_device: Device,

    /// Decoder batch
    #[arg(long, default_value_t = 1)]
    pub decoder_batch: usize,

    /// Processor device (for pre/post processing)
    #[arg(long, global = true, default_value = "cpu")]
    pub processor_device: Device,

    /// num dry run
    #[arg(long, global = true, default_value_t = 0)]
    pub num_dry_run: usize,

    /// trt_max_workspace_size
    #[arg(long, global = true, default_value_t = 3221225472)]
    pub trt_max_workspace_size: usize,
}

pub fn config(args: &EfficientSam3Args) -> Result<Config> {
    let config = match args.variant {
        EfficientSam3Variant::EvM => Config::efficient_sam3_ev_m(),
        EfficientSam3Variant::RvM => Config::efficient_sam3_rv_m(),
        EfficientSam3Variant::TvM => Config::efficient_sam3_tv_m(),
    };

    let config = if let Some(dir) = &args.model_dir {
        config
            .with_visual_encoder_file(dir.join("vision-encoder.onnx").to_string_lossy())
            .with_textual_encoder_file(dir.join("text-encoder.onnx").to_string_lossy())
            .with_decoder_file(dir.join("decoder.onnx").to_string_lossy())
    } else {
        config
    };

    let config = config
        .with_visual_encoder_batch_min_opt_max(1, args.visual_encoder_batch, 2)
        .with_textual_encoder_batch_min_opt_max(1, args.textual_encoder_batch, 2)
        .with_decoder_batch_min_opt_max(1, args.decoder_batch, 2)
        .with_visual_encoder_device(args.visual_encoder_device)
        .with_visual_encoder_dtype(args.visual_encoder_dtype)
        .with_textual_encoder_device(args.textual_encoder_device)
        .with_textual_encoder_dtype(args.textual_encoder_dtype)
        .with_decoder_device(args.decoder_device)
        .with_decoder_dtype(args.decoder_dtype)
        .with_num_dry_run_all(args.num_dry_run)
        .with_image_processor_device(args.processor_device)
        .with_tensorrt_max_workspace_size_all(args.trt_max_workspace_size);

    Ok(config)
}
