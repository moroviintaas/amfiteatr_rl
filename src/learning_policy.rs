use tch::nn::VarStore;
use sztorm::agent::AgentTrajectory;
use sztorm::error::SztormError;
use sztorm::protocol::DomainParameters;
use sztorm::state::agent::{InformationSet, ScoringInformationSet};

pub trait LearningNetworkPolicy<DP: DomainParameters, InfoSet: ScoringInformationSet<DP>> {
    type Network;
    type TrainConfig;

    fn network(&self) -> &Self::Network;
    fn network_mut(&mut self) -> &mut Self::Network;
    fn var_store(&self) -> &VarStore;
    fn var_store_mut(&mut self) -> &mut VarStore;

    fn batch_train_on_universal_rewards(&mut self, trajectories: &[AgentTrajectory<DP, InfoSet>], train_config: &Self::TrainConfig) -> Result<(), SztormError<DP>>;



}