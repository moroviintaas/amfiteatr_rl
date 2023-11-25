mod tensor_repr;
pub use tensor_repr::*;

use tch::TchError;
use thiserror::Error;
use amfi::error::AmfiError;
use amfi::domain::DomainParameters;


#[derive(Error, Debug)]
pub enum AmfiRLError<DP: DomainParameters>{
    #[error("Basic amfi error: {0}")]
    Amfi(AmfiError<DP>),
    #[error("Torch error: {error} in context: {context:}")]
    Torch{
        error: TchError,
        context: String
    },
    #[error("Tensor representation: {0}")]
    TensorRepresentation(TensorRepresentationError),


}

impl<DP: DomainParameters> From<TchError> for AmfiRLError<DP>{
    fn from(value: TchError) -> Self {
        Self::Torch{
            error: value,
            context: String::from("unspecified")
        }
    }
}

impl<DP: DomainParameters> From<AmfiError<DP>> for AmfiRLError<DP>{
    fn from(value: AmfiError<DP>) -> Self {
        Self::Amfi(value)
    }
}

impl<DP: DomainParameters> From<AmfiRLError<DP>> for AmfiError<DP>{
    fn from(value: AmfiRLError<DP>) -> Self {
        match value{
            AmfiRLError::Amfi(n) => n,
            any => AmfiError::Custom(format!("{:?}", any))
        }
    }
}