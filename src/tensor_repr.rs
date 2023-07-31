use tch::Tensor;
use sztorm::Action;
use sztorm::error::{ConvertError};

pub trait TensorBuilder<T>: Send{
    type Error: std::error::Error;
    fn build_tensor(&self, t: &T) -> Result<Tensor, Self::Error>;
}

pub trait ConvStateToTensor<T>: Send{
    fn make_tensor(&self, t: &T) -> Tensor;
}

pub trait TensorInterpreter<T>: Send{
    type Error: std::error::Error;
    fn interpret_tensor(&self, tensor: &Tensor) -> Result<T, Self::Error>;
}

pub trait ActionTensor: Action{

    fn to_tensor(&self) -> Tensor;
    fn try_from_tensor(t: &Tensor) -> Result<Self, ConvertError>;
}