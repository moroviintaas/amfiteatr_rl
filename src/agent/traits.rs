use amfi::agent::{
    AutomaticAgent,
    AutomaticAgentRewarded,
    PolicyAgent,
    ReseedAgent,
    ScoringInformationSet,
    StatefulAgent,
    TracingAgent};
use amfi::domain::DomainParameters;
use crate::LearningNetworkPolicy;

pub trait NetworkLearningAgent<DP: DomainParameters>: AutomaticAgentRewarded<DP>  + PolicyAgent<DP> + TracingAgent<DP, <Self as StatefulAgent<DP>>::InfoSetType>
    where  <Self as PolicyAgent<DP>>::Policy: LearningNetworkPolicy<DP>,
    <Self as StatefulAgent<DP>>::InfoSetType: ScoringInformationSet<DP>
{
}

impl <DP: DomainParameters, T: AutomaticAgentRewarded<DP>  + PolicyAgent<DP>
+ TracingAgent<DP, <Self as StatefulAgent<DP>>::InfoSetType>>
NetworkLearningAgent<DP> for T
where <T as PolicyAgent<DP>>::Policy: LearningNetworkPolicy<DP>,
<T as StatefulAgent<DP>>::InfoSetType: ScoringInformationSet<DP>
{
}


pub trait TestingAgent<DP: DomainParameters>: AutomaticAgent<DP>  + PolicyAgent<DP>
 + TracingAgent<DP, <Self as StatefulAgent<DP>>::InfoSetType>
where <Self as StatefulAgent<DP>>::InfoSetType: ScoringInformationSet<DP>{}

impl <DP: DomainParameters, T: AutomaticAgent<DP>  + PolicyAgent<DP>
+ TracingAgent<DP, <Self as StatefulAgent<DP>>::InfoSetType>>

TestingAgent<DP> for T
where <T as StatefulAgent<DP>>::InfoSetType: ScoringInformationSet<DP>
{}

pub trait RlModelAgent<DP: DomainParameters, Seed>: AutomaticAgentRewarded<DP> + ReseedAgent<DP, Seed>
{

}