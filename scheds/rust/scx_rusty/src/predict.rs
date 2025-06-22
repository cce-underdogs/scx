use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{ops::sigmoid, var_builder::VarBuilder, VarMap};

use crate::model::ResNetModel;
use log::debug;

const INPUT_DIM: usize = 3;
const HIDDEN_DIM: usize = 10;
const OUTPUT_DIM: usize = 1;

pub struct Predictor {
    model: ResNetModel,
    device: Device,
}

impl Predictor {
    pub fn new(weights_path: &str) -> Result<Self> {
        let device = Device::Cpu;
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&mut varmap, DType::F32, &device);
        let model = ResNetModel::new(&vb, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)?;

        varmap.load(weights_path)?;
        Ok(Self { model, device })
    }

    pub fn predict(&self, input_raw: &[f32]) -> Result<(u32, f32)> {
        let input = Tensor::from_vec(input_raw.to_vec(), (1, INPUT_DIM), &self.device)?;
        let logits = self.model.forward(&input)?;
        let prob = sigmoid(&logits)?;
        let predicted = prob.ge(0.5)?.to_dtype(DType::U32)?;

        let class = predicted.squeeze(0)?.squeeze(0)?.to_scalar::<u32>()?;
        let confidence = prob.squeeze(0)?.squeeze(0)?.to_scalar::<f32>()?;
        Ok((class, confidence))
    }
}
