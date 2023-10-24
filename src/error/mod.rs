use tch::TchError;
use thiserror::Error;
use amfi::error::AmfiError;
use amfi::domain::DomainParameters;


#[derive(Error, Debug)]
pub enum AmfiRLError<DP: DomainParameters>{
    #[error("Basic amfi error: {0}")]
    Amfi(AmfiError<DP>),
    #[error("Torch error: {0}")]
    Torch(TchError),

}

impl<DP: DomainParameters> From<TchError> for AmfiRLError<DP>{
    fn from(value: TchError) -> Self {
        Self::Torch(value)
    }
}

impl<DP: DomainParameters> From<AmfiError<DP>> for AmfiRLError<DP>{
    fn from(value: AmfiError<DP>) -> Self {
        Self::Amfi(value)
    }
}