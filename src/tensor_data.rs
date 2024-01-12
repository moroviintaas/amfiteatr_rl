use std::error::Error;
use std::fmt::{Debug, Display};
use tch::{Tensor};
use amfi_core::domain::{Action, Reward};
use amfi_core::error::{ConvertError};
use crate::error::TensorRepresentationError;


pub trait ConvStateToTensor<T>: Send{
    fn make_tensor(&self, t: &T) -> Tensor;
}



pub trait ConversionToTensor: Send + Default{
    fn desired_shape(&self) -> &[i64];

    fn desired_shape_flatten(&self) -> i64{
        self.desired_shape().iter().product()
    }
}

pub trait ConvertToTensor<W: ConversionToTensor> : Debug{
    fn try_to_tensor(&self, way: &W) -> Result<Tensor, TensorRepresentationError>;

    fn to_tensor(&self, way: &W) -> Tensor{
        self.try_to_tensor(way).unwrap()
    }
    fn try_to_tensor_flat(&self, way: &W) -> Result<Tensor, TensorRepresentationError>{
        let t1 = self.try_to_tensor(way)?;
        t1.f_flatten(0, -1).map_err(|e|{
            TensorRepresentationError::Torch {
                error: e,
                context: format!("Flattening tensor {t1:?} from information set: {:?}", self)
            }
        })
    }

    fn to_tensor_flat(&self, way: &W) -> Tensor{
        let t1 = self.to_tensor(way);
        //let dim = t1.dim() as i64;
        t1.flatten(0, -1)
    }
    fn tensor_shape(way: &W) -> &[i64]{
        way.desired_shape()
    }
    fn tensor_length_flatten(way: &W) -> i64{
        way.desired_shape().iter().product()
    }
}

impl<W: ConversionToTensor, T: ConvertToTensor<W>> ConvertToTensor<W> for Box<T>{
    fn try_to_tensor(&self, way: &W) -> Result<Tensor, TensorRepresentationError> {
        self.as_ref().try_to_tensor(way)
    }
}

pub trait ConversionFromTensor: Send{
    fn expected_input_shape() -> &'static[i64];
}

pub trait TryConvertFromTensor<W: ConversionFromTensor>{
    type ConvertError: Error;
    fn try_from_tensor(tensor: &Tensor, way: &W) -> Result<Self, Self::ConvertError> where Self: Sized;
}

pub trait ConvertToTensorD<W: ConversionToTensor>: ConvertToTensor<W> + Display + Debug{}
impl<W: ConversionToTensor, T: ConvertToTensor<W> + Display + Debug> ConvertToTensorD<W> for T{}


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

