use std::fmt::Debug;
use std::marker::PhantomData;
use tch::Kind::Float;
use tch::nn::Optimizer;
use sztorm::agent::{AgentTrajectory, Policy};
use sztorm::protocol::DomainParameters;
use sztorm::RewardSource;
use sztorm::state::agent::{InformationSet, ScoringInformationSet};
use crate::experiencing_policy::SelfExperiencingPolicy;
use crate::tensor_repr::{TensorBuilder, TensorInterpreter};
use crate::torch_net::{A2CNet};


pub struct ActorCriticPolicy<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug,
    StateConverter: TensorBuilder<InfoSet>,
    ActInterpreter: TensorInterpreter<Option<DP::ActionType>>
> {
    network: A2CNet,
    #[allow(dead_code)]
    optimizer: Optimizer,
    _dp: PhantomData<DP>,
    _is: PhantomData<InfoSet>,
    state_converter: StateConverter,
    action_interpreter: ActInterpreter

}

impl<
    DP: DomainParameters,
    InfoSet: ScoringInformationSet<DP> + Debug,
    StateConverter: TensorBuilder<InfoSet>,
    ActInterpreter: TensorInterpreter<Option<DP::ActionType>>
> ActorCriticPolicy<DP, InfoSet, StateConverter, ActInterpreter>{
    pub fn new(network: A2CNet,
               optimizer: Optimizer,
               state_converter: StateConverter,
               action_interpreter: ActInterpreter) -> Self{
        Self{network, optimizer, state_converter, action_interpreter, _dp: Default::default(), _is: Default::default()}
    }

    pub fn batch_train(&mut self, trajectories: &[AgentTrajectory<DP, InfoSet>], gamma: f64, reward_source: RewardSource){
        //let state_tensor = trajectories.iter().

        // states
        // rewards -> s_returns

        for t in trajectories{

        }
        todo!();
    }
}

impl<DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug,
    TB: TensorBuilder<InfoSet>,
    ActInterpreter: TensorInterpreter<Option<DP::ActionType>>
> Policy<DP> for ActorCriticPolicy<DP, InfoSet, TB, ActInterpreter>{
    type StateType = InfoSet;

    fn select_action(&self, state: &Self::StateType) -> Option<DP::ActionType> {
        let state_tensor = self.state_converter.build_tensor(state)
            .unwrap_or_else(|_| panic!("Failed converting state to Tensor: {:?}", state));
        let out = tch::no_grad(|| (self.network.net())(&state_tensor));
        let actor = out.actor;
        //somewhen it may be changed with temperature
        let probs = actor.softmax(-1, Float);
        let atensor = probs.multinomial(1, true);
        self.action_interpreter.interpret_tensor(&atensor)
            .expect("Failed converting tensor to action")

    }
}


impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug,
    TB: TensorBuilder<InfoSet>,
    ActInterpreter: TensorInterpreter<Option<DP::ActionType>>> SelfExperiencingPolicy<DP> for ActorCriticPolicy<DP, InfoSet, TB, ActInterpreter>
where DP::ActionType: From<i64>{
    type PolicyUpdateError = tch::TchError;

    fn select_action_and_collect_experience(&mut self) -> Option<DP::ActionType> {
        todo!()
    }


    fn apply_experience(&mut self) -> Result<(), Self::PolicyUpdateError> {
        todo!()
    }
}
