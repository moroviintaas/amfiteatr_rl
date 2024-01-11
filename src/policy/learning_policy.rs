use tch::nn::VarStore;
use tch::Tensor;
use amfi_core::agent::{AgentTraceStep, Trajectory, Policy, EvaluatedInformationSet};

use amfi_core::domain::DomainParameters;
use crate::error::AmfiRLError;
use crate::tensor_repr::FloatTensorReward;


pub trait DiscountAggregator{
    fn discount_factor(&self) -> f64;
}

pub trait LearningNetworkPolicy<DP: DomainParameters> : Policy<DP>
where <Self as Policy<DP>>::InfoSetType: EvaluatedInformationSet<DP>
{
    type Network;
    type TrainConfig;

    fn network(&self) -> &Self::Network;
    fn network_mut(&mut self) -> &mut Self::Network;
    fn var_store(&self) -> &VarStore;
    fn var_store_mut(&mut self) -> &mut VarStore;

    fn switch_explore(&mut self, enabled: bool);

    fn config(&self) -> &Self::TrainConfig;
    fn train_on_trajectories<R: Fn(&AgentTraceStep<DP, <Self as Policy<DP>>::InfoSetType>) -> Tensor>(
        &mut self,
        trajectories: &[Trajectory<AgentTraceStep<DP, <Self as Policy<DP>>::InfoSetType>>],
        reward_f: R,
    ) -> Result<(), AmfiRLError<DP>>;

    fn train_on_trajectories_env_reward(&mut self,
                                        trajectories: &[Trajectory<AgentTraceStep<DP, <Self as Policy<DP>>::InfoSetType>>]) -> Result<(), AmfiRLError<DP>>
    where <DP as DomainParameters>::UniversalReward: FloatTensorReward{

        self.train_on_trajectories(trajectories,  |step| step.step_universal_reward().to_tensor())
    }

    fn train_on_trajectories_info_set_rewards(&mut self,
                                              trajectories: &[Trajectory<AgentTraceStep<DP, <Self as Policy<DP>>::InfoSetType>>],
                                              ) -> Result<(), AmfiRLError<DP>>
    where <<Self as Policy<DP>>::InfoSetType as EvaluatedInformationSet<DP>>::RewardType: FloatTensorReward{

        self.train_on_trajectories(trajectories,  |step| step.step_subjective_reward().to_tensor())
    }

}