use tch::Tensor;
use amfi_core::demo::{DemoAction, DemoInfoSet};
use amfi_core::error::ConvertError;
use crate::error::TensorRepresentationError;
use crate::tensor_repr::{ActionTensor, ConvertToTensor, WayToTensor};

#[derive(Default, Copy, Clone, Debug)]
pub struct DemoInfoSetWay{}



impl WayToTensor for DemoInfoSetWay{
    fn desired_shape(&self) -> &[i64] {
        &[1]
    }
}

impl ConvertToTensor<DemoInfoSetWay> for DemoInfoSet{
    fn try_to_tensor(&self, _way: &DemoInfoSetWay) -> Result<Tensor, TensorRepresentationError> {
        Ok(Tensor::from_slice(&[1.0]))
    }
}

impl ActionTensor for DemoAction{
    fn to_tensor(&self) -> Tensor {
        Tensor::from_slice(&[self.0 as f32])
    }

    fn try_from_tensor(t: &Tensor) -> Result<Self, ConvertError> {
        let v: Vec<f32> = Vec::try_from(t).unwrap();
        Ok(DemoAction{0: v[0] as u8})
    }
}