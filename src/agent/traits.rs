use amfi::agent::*;
use amfi::domain::DomainParameters;
use crate::LearningNetworkPolicy;

pub trait NetworkLearningAgent<DP: DomainParameters>: AutomaticAgentRewarded<DP>  + PolicyAgent<DP> + TracingAgent<DP, <Self as StatefulAgent<DP>>::InfoSetType>
    where  <Self as PolicyAgent<DP>>::Policy: LearningNetworkPolicy<DP>,
    <Self as StatefulAgent<DP>>::InfoSetType: EvaluatedInformationSet<DP>
{
}

impl <DP: DomainParameters, T: AutomaticAgentRewarded<DP>  + PolicyAgent<DP>
+ TracingAgent<DP, <Self as StatefulAgent<DP>>::InfoSetType>>
NetworkLearningAgent<DP> for T
where <T as PolicyAgent<DP>>::Policy: LearningNetworkPolicy<DP>,
<T as StatefulAgent<DP>>::InfoSetType: EvaluatedInformationSet<DP>
{
}


pub trait TestingAgent<DP: DomainParameters>: AutomaticAgent<DP>  + PolicyAgent<DP>
 + TracingAgent<DP, <Self as StatefulAgent<DP>>::InfoSetType>
where <Self as StatefulAgent<DP>>::InfoSetType: EvaluatedInformationSet<DP>{}

impl <DP: DomainParameters, T: AutomaticAgent<DP>  + PolicyAgent<DP>
+ TracingAgent<DP, <Self as StatefulAgent<DP>>::InfoSetType>>

TestingAgent<DP> for T
where <T as StatefulAgent<DP>>::InfoSetType: EvaluatedInformationSet<DP>
{}




pub trait RlModelAgent<DP: DomainParameters, Seed, IS: EvaluatedInformationSet<DP>>:
    AutomaticAgentRewardedAndEvaluated<DP, Assessment= <IS as EvaluatedInformationSet<DP>>::RewardType>
    + MultiEpisodeAutoAgentRewarded<DP, Seed>
    + PolicyAgent<DP> + StatefulAgent<DP, InfoSetType=IS>
    + TracingAgent<DP, AgentTraceStep<DP, IS>>
    + Send

where <Self as PolicyAgent<DP>>::Policy: LearningNetworkPolicy<DP>,
{}




impl<
    DP: DomainParameters,
    Seed,
    IS: EvaluatedInformationSet<DP>,
    T: AutomaticAgentRewardedAndEvaluated<DP, Assessment= <IS as EvaluatedInformationSet<DP>>:: RewardType>
        + MultiEpisodeAutoAgentRewarded<DP, Seed>
        + PolicyAgent<DP> + StatefulAgent<DP, InfoSetType=IS>
        + TracingAgent<DP, AgentTraceStep<DP, IS>>
        + Send

> RlModelAgent<DP, Seed, IS> for T
where <Self as PolicyAgent<DP>>::Policy: LearningNetworkPolicy<DP>,{

}