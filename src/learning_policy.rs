use tch::nn::VarStore;
use sztorm::agent::{AgentTrajectory, Policy, ScoringInformationSet};

use sztorm::domain::DomainParameters;
use crate::error::SztormRLError;

pub trait LearningNetworkPolicy<DP: DomainParameters> : Policy<DP>
where <Self as Policy<DP>>::InfoSetType: ScoringInformationSet<DP>
{
    type Network;
    type TrainConfig;

    fn network(&self) -> &Self::Network;
    fn network_mut(&mut self) -> &mut Self::Network;
    fn var_store(&self) -> &VarStore;
    fn var_store_mut(&mut self) -> &mut VarStore;

    fn batch_train_on_universal_rewards(&mut self, trajectories: &[AgentTrajectory<DP, <Self as Policy<DP>>::InfoSetType>]) -> Result<(), SztormRLError<DP>>;
    fn config(&self) -> &Self::TrainConfig;

}