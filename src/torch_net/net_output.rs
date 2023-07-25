use tch::Tensor;

pub trait NetOutput{}

pub struct TensorA2C{
    pub critic: Tensor,
    pub actor: Tensor
}

impl NetOutput for Tensor{}
impl NetOutput for (Tensor, Tensor){}
impl NetOutput for TensorA2C{}