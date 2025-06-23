use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{ops::sigmoid, var_builder::VarBuilder, VarMap};

use crate::model::ResNetModel;
use log::debug;
use serde::Deserialize;
use std::fs::File;

const INPUT_DIM: usize = 6;
const HIDDEN_DIM: usize = 10;
const OUTPUT_DIM: usize = 1;
#[derive(serde::Deserialize)]
struct NormParams {
    min: Vec<f32>,
    max: Vec<f32>,
}

pub struct Predictor {
    model: ResNetModel,
    device: Device,
    norm: NormParams,
}

impl Predictor {
    fn normalize_input(input: Vec<f32>, min: &[f32], max: &[f32]) -> Vec<f32> {
        input
            .iter()
            .zip(min.iter().zip(max.iter()))
            .map(|(x, (&min, &max))| {
                if (max - min).abs() < 1e-6 {
                    0.0 // 避免除以 0
                } else {
                    (x - min) / (max - min)
                }
            })
            .collect()
    }

    pub fn new(weights_path: &str, norm_path: &str) -> Result<Self> {
        let device = Device::Cpu;
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&mut varmap, DType::F32, &device);
        let model = ResNetModel::new(&vb, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)?;

        let file = File::open(norm_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to open {}: {e}", norm_path)))?;
        let norm: NormParams = serde_json::from_reader(file)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to parse norm.json: {e}")))?;

        varmap.load(weights_path)?;
        Ok(Self {
            model,
            device,
            norm,
        })
    }

    pub fn predict(&self, input_raw: &[f32]) -> Result<(u32, f32)> {
        let input_norm = Self::normalize_input(input_raw.to_vec(), &self.norm.min, &self.norm.max);
        let input = Tensor::from_vec(input_norm, (1, INPUT_DIM), &self.device)?;
        let logits = self.model.forward(&input)?;
        let prob = sigmoid(&logits)?;
        let predicted = prob.ge(0.5)?.to_dtype(DType::U32)?;

        let class = predicted.squeeze(0)?.squeeze(0)?.to_scalar::<u32>()?;
        let confidence = prob.squeeze(0)?.squeeze(0)?.to_scalar::<f32>()?;
        Ok((class, confidence))
    }
}
