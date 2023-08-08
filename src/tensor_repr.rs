use tch::{Kind, Tensor};
use sztorm::{Action, Reward};
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

pub trait TensorReward: Reward{
    //type Dims: IntList;
    fn kind() -> Kind;
    fn to_tensor(&self) -> Tensor;
    //fn shape(&self) -> Dims;
    fn shape() -> Vec<i64>;
}

macro_rules! impl_reward_std_f {
    ($($x: ty), +) => {
        $(
        impl TensorReward for $x{
            fn kind() -> Kind {
                Kind::Float
            }

            fn to_tensor(&self) -> Tensor {
                let s = [*self;1];
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


impl TensorReward for i64{
    fn kind() -> Kind {
        Kind::Int64
    }

    fn to_tensor(&self) -> Tensor {
        let s = [*self;1];
        Tensor::from_slice(&s[..])

    }

    fn shape() -> Vec<i64> {
        vec![1]
    }
}

