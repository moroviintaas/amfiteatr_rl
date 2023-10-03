use tch::TchError;
use thiserror::Error;
use sztorm::error::SztormError;
use sztorm::domain::DomainParameters;


#[derive(Error, Debug)]
pub enum SztormRLError<DP: DomainParameters>{
    #[error("Basic sztorm error: {0}")]
    Sztorm(SztormError<DP>),
    #[error("Torch error: {0}")]
    Torch(TchError),

}

impl<DP: DomainParameters> From<TchError> for SztormRLError<DP>{
    fn from(value: TchError) -> Self {
        Self::Torch(value)
    }
}

impl<DP: DomainParameters> From<SztormError<DP>> for SztormRLError<DP>{
    fn from(value: SztormError<DP>) -> Self {
        Self::Sztorm(value)
    }
}