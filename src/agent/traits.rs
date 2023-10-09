use sztorm::agent::{AutomaticAgent, AutomaticAgentRewarded, PolicyAgent, ScoringInformationSet, StatefulAgent, TracingAgent};
use sztorm::domain::DomainParameters;
use crate::LearningNetworkPolicy;

pub trait NetworkLearningAgent<DP: DomainParameters>: AutomaticAgentRewarded<DP>  + PolicyAgent<DP> + TracingAgent<DP, <Self as StatefulAgent<DP>>::State>
    where  <Self as PolicyAgent<DP>>::Policy: LearningNetworkPolicy<DP>,
    <Self as StatefulAgent<DP>>::State: ScoringInformationSet<DP>
{
}

impl <DP: DomainParameters, T: AutomaticAgentRewarded<DP>  + PolicyAgent<DP>
+ TracingAgent<DP, <Self as StatefulAgent<DP>>::State>>
NetworkLearningAgent<DP> for T
where <T as PolicyAgent<DP>>::Policy: LearningNetworkPolicy<DP>,
<T as StatefulAgent<DP>>::State: ScoringInformationSet<DP>
{
}


pub trait TestingAgent<DP: DomainParameters>: AutomaticAgent<DP>  + PolicyAgent<DP>
 + TracingAgent<DP, <Self as StatefulAgent<DP>>::State>
where <Self as StatefulAgent<DP>>::State: ScoringInformationSet<DP>{}

impl <DP: DomainParameters, T: AutomaticAgent<DP>  + PolicyAgent<DP>
+ TracingAgent<DP, <Self as StatefulAgent<DP>>::State>>

TestingAgent<DP> for T
where <T as StatefulAgent<DP>>::State: ScoringInformationSet<DP>
{}