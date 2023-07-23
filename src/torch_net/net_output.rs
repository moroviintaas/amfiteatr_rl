use tch::Tensor;

pub trait NetOutput{}

impl NetOutput for Tensor{}
impl NetOutput for (Tensor, Tensor){}