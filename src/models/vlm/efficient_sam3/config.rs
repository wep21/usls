use crate::{Config, Task};

const EFFICIENT_SAM3_RELEASE: &str =
    "https://github.com/wep21/assets/releases/download/efficient-sam3";

///
/// > # EfficientSAM3
/// >
/// > Lightweight SAM3 image segmentation models using EfficientViT/RepViT/TinyViT
/// > image encoders and MobileCLIP-S0 text encoding.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [SimonZeng7108/efficientsam3](https://github.com/SimonZeng7108/efficientsam3)
/// > - **Release**: [v0.4.0-efficientsam3ft-20260611](https://github.com/SimonZeng7108/efficientsam3/releases/tag/v0.4.0-efficientsam3ft-20260611)
/// > - **ONNX Models**: [wep21/assets release](https://github.com/wep21/assets/releases/tag/efficient-sam3)
/// >
/// > # Model Variants
/// >
/// > - **efficient-sam3-ev-m**: EfficientViT-B1 + MobileCLIP-S0
/// > - **efficient-sam3-rv-m**: RepViT-M1.1 + MobileCLIP-S0
/// > - **efficient-sam3-tv-m**: TinyViT-11M + MobileCLIP-S0
///
impl Config {
    /// EfficientSAM3 image configuration with MobileCLIP-S0 text tokenization.
    ///
    /// EfficientSAM3 and SAM3-LiteText use the MobileCLIP/OpenAI CLIP BPE
    /// tokenizer with a 16-token context for the released S0 models.
    fn efficient_sam3(asset_prefix: &str) -> Self {
        Self::sam3()
            .with_name("efficient-sam3")
            .with_task(Task::Sam3Image)
            .with_resize_mode_type(crate::ResizeModeType::FitExact)
            .with_tokenizer_file("clip/tokenizer.json")
            .with_tokenizer_config_file("clip/tokenizer_config.json")
            .with_special_tokens_map_file("clip/special_tokens_map.json")
            .with_config_file("clip/config.json")
            .with_model_max_length(16)
            .with_visual_encoder_file(format!(
                "{EFFICIENT_SAM3_RELEASE}/{asset_prefix}-vision-encoder.onnx"
            ))
            .with_textual_encoder_file(format!(
                "{EFFICIENT_SAM3_RELEASE}/{asset_prefix}-text-encoder.onnx"
            ))
            .with_decoder_file(format!(
                "{EFFICIENT_SAM3_RELEASE}/{asset_prefix}-decoder.onnx"
            ))
            .with_visual_encoder_batch_min_opt_max(1, 1, 4)
            .with_textual_encoder_batch_min_opt_max(1, 1, 8)
            .with_decoder_batch_min_opt_max(1, 1, 4)
            .with_decoder_ixx(6, 1, (1, 1, 8))
            .with_decoder_ixx(7, 1, (1, 1, 8))
    }

    /// EfficientSAM3 EV-M: EfficientViT-B1 vision encoder + MobileCLIP-S0 text encoder.
    pub fn efficient_sam3_ev_m() -> Self {
        Self::efficient_sam3("efficient-sam3-ev-m").with_name("efficient-sam3-ev-m")
    }

    /// EfficientSAM3 RV-M: RepViT-M1.1 vision encoder + MobileCLIP-S0 text encoder.
    pub fn efficient_sam3_rv_m() -> Self {
        Self::efficient_sam3("efficient-sam3-rv-m").with_name("efficient-sam3-rv-m")
    }

    /// EfficientSAM3 TV-M: TinyViT-11M vision encoder + MobileCLIP-S0 text encoder.
    pub fn efficient_sam3_tv_m() -> Self {
        Self::efficient_sam3("efficient-sam3-tv-m").with_name("efficient-sam3-tv-m")
    }
}
