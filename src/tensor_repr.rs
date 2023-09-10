use std::error::Error;
use std::fmt::{Debug, Display};
use tch::{Tensor};
use sztorm::{Action, Reward};
use sztorm::error::{ConvertError};
use sztorm::protocol::DomainParameters;
use sztorm::state::agent::InformationSet;

pub trait TensorBuilder<T>: Send{
    type Error: std::error::Error;
    fn build_tensor(&self, t: &T) -> Result<Tensor, Self::Error>;
}

pub trait ConvStateToTensor<T>: Send{
    fn make_tensor(&self, t: &T) -> Tensor;
}

pub trait WayToTensor: Send + Default{
    fn desired_shape() -> &'static[i64];
}

pub trait ConvertToTensor<W: WayToTensor>{
    fn to_tensor(&self, way: &W) -> Tensor;
}

impl<W: WayToTensor, T: ConvertToTensor<W>> ConvertToTensor<W> for Box<T>{
    fn to_tensor(&self, way: &W) -> Tensor {
        self.as_ref().to_tensor(way)
    }
}

pub trait WayFromTensor: Send{
    fn expected_input_shape() -> &'static[i64];
}

pub trait TryConvertFromTensor<W: WayFromTensor>{
    type ConvertError: Error;
    fn try_from_tensor(tensor: &Tensor, way: &W) -> Result<Self, Self::ConvertError> where Self: Sized;
}

pub trait ConvertToTensorD<W: WayToTensor>: ConvertToTensor<W> + Display + Debug{}
impl<W: WayToTensor, T: ConvertToTensor<W> + Display + Debug> ConvertToTensorD<W> for T{}
/*
impl<DP: DomainParameters, S: InformationSet<DP>, T: ConvStateToTensor<S>> ConvStateToTensor<Box<S>> for T{
    fn make_tensor(&self, t: &Box<S>) -> Tensor {
        self.make_tensor(t.as_ref())
    }
}

 */



pub trait TensorInterpreter<T>: Send{
    type Error: std::error::Error;
    fn interpret_tensor(&self, tensor: &Tensor) -> Result<T, Self::Error>;
}

pub trait ActionTensor: Action{

    fn to_tensor(&self) -> Tensor;
    fn try_from_tensor(t: &Tensor) -> Result<Self, ConvertError>;
}

pub trait FloatTensorReward: Reward{
    //type Dims: IntList;
    fn to_tensor(&self) -> Tensor;
    //fn shape(&self) -> Dims;
    fn shape() -> Vec<i64>;
    fn total_size() -> i64{
        Self::shape().iter().fold(0, |acc, x| acc+x)
    }
}

macro_rules! impl_reward_std_f {
    ($($x: ty), +) => {
        $(
        impl FloatTensorReward for $x{

            fn to_tensor(&self) -> Tensor {
                let s = [*self as f32;1];
                Tensor::from_slice(&s[..])

            }

            fn shape() -> Vec<i64> {
                vec![1]
            }
        }

        )*

    }
}

impl_reward_std_f![f32, f64];


impl FloatTensorReward for i64{

    fn to_tensor(&self) -> Tensor {
        let s = [*self as f32;1];
        Tensor::from_slice(&s[..])

    }

    fn shape() -> Vec<i64> {
        vec![1]
    }
}

impl FloatTensorReward for i32{

    fn to_tensor(&self) -> Tensor {
        let s = [*self as f32];
        Tensor::from_slice(&s[..])

    }

    fn shape() -> Vec<i64> {
        vec![1]
    }
}

