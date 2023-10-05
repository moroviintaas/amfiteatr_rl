use std::error::Error;
use sztorm::agent::{AgentTrajectory, Policy};
use sztorm::domain::DomainParameters;
use sztorm::state::agent::ScoringInformationSet;


pub trait SelfExperiencingPolicy<DP:  DomainParameters>{
    type PolicyUpdateError: Error;
    fn select_action_and_collect_experience(&mut self) -> Option<DP::ActionType>;
    /*fn policy_update(&mut self, traces: &Vec<GameTrace<DP, <Self as Policy<DP>>::StateType>>)
        -> Result<(), Self::PolicyUpdateError>;

     */

    fn apply_experience(&mut self) -> Result<(), Self::PolicyUpdateError>;
}

pub trait UpdatablePolicy<DP:  DomainParameters>: Policy<DP>
where <Self as Policy<DP>>::InfoSetType: ScoringInformationSet<DP>{
    type PolicyUpdateError: Error;
    fn policy_update(&mut self, traces: &[AgentTrajectory<DP, <Self as Policy<DP>>::InfoSetType>])
        -> Result<(), Self::PolicyUpdateError>;

}