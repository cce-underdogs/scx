use candle_core::{Result, Tensor};
use candle_nn::var_builder::VarBuilder;
use candle_nn::{Linear, Module};

pub struct ResNetBlock {
    linear1: Linear,
    linear2: Linear,
}

impl ResNetBlock {
    fn new(vb: &VarBuilder, dim: usize) -> Result<Self> {
        let linear1 = candle_nn::linear(dim, dim, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(dim, dim, vb.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.linear1.forward(x)?.relu()?;
        let h = self.linear2.forward(&h)?;
        x.add(&h)?.relu()
    }
}

pub struct ResNetModel {
    input: Linear,
    block: ResNetBlock,
    output: Linear,
}

impl ResNetModel {
    pub fn new(
        vb: &VarBuilder,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
    ) -> Result<Self> {
        let input = candle_nn::linear(input_dim, hidden_dim, vb.pp("input"))?;
        let block = ResNetBlock::new(&vb.pp("block"), hidden_dim)?;
        let output = candle_nn::linear(hidden_dim, output_dim, vb.pp("output"))?;
        Ok(Self {
            input,
            block,
            output,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.input.forward(x)?.relu()?;
        let h = self.block.forward(&h)?;
        self.output.forward(&h)
    }
}
