use std::fmt::Debug;
use std::marker::PhantomData;
use tch::Kind::Float;
use tch::kind::FLOAT_CPU;
use tch::nn::Optimizer;
use tch::Tensor;
use sztorm::agent::{AgentTrajectory, Policy};
use sztorm::error::SztormError;
use sztorm::protocol::DomainParameters;
use sztorm::RewardSource;
use sztorm::state::agent::{InformationSet, ScoringInformationSet};
use crate::experiencing_policy::SelfExperiencingPolicy;
use crate::tensor_repr::{ActionTensor, ConvStateToTensor, TensorInterpreter};
use crate::torch_net::{A2CNet};


pub struct ActorCriticPolicy<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug,
    StateConverter: ConvStateToTensor<InfoSet>,
    //ActInterpreter: TensorInterpreter<Option<DP::ActionType>>
> {
    network: A2CNet,
    #[allow(dead_code)]
    optimizer: Optimizer,
    _dp: PhantomData<DP>,
    _is: PhantomData<InfoSet>,
    state_converter: StateConverter,
    //action_interpreter: ActInterpreter

}

impl<
    DP: DomainParameters,
    InfoSet: ScoringInformationSet<DP> + Debug,
    StateConverter: ConvStateToTensor<InfoSet>,
    //ActInterpreter: TensorInterpreter<Option<DP::ActionType>>
> ActorCriticPolicy<DP, InfoSet, StateConverter,
    /*ActInterpreter*/
>
where <DP as DomainParameters>::ActionType: ActionTensor{
    pub fn new(network: A2CNet,
               optimizer: Optimizer,
               state_converter: StateConverter,
               /*action_interpreter: ActInterpreter*/
    ) -> Self{
        Self{network, optimizer, state_converter,
            //action_interpreter,
            _dp: Default::default(), _is: Default::default()}
    }

    /*
    pub fn batch_train(&mut self, trajectories: &[AgentTrajectory<DP, InfoSet>], gamma: f64, reward_source: RewardSource)
        -> SztormError<DP>
    where for<'a> &'a <DP as DomainParameters>::UniversalReward: Into<Tensor>,
    for<'a> &'a <InfoSet as ScoringInformationSet<DP>>::RewardType: Into<Tensor>{



        for t in trajectories{
            if t.list().is_empty(){
                continue;
            }
            let state_tensor_results: Vec<Tensor> = t.list().iter().map(|step|{
                self.state_converter.make_tensor(step.step_state())
            }).collect();

            let final_score_t: Tensor = match reward_source{
                RewardSource::Env => t.list().last().unwrap().universal_score_after().into(),
                RewardSource::Agent => t.list().last().unwrap().subjective_score_after().into(),
            };
            let discounted_rewards = {
                let mut r = Tensor::zeros([state_tensor_results.len() as i64 + 1, 1 ], FLOAT_CPU);
                for s in (0..state_tensor_results.len()).rev(){
                    /*
                    let step_reward = match reward_source{
                        //
                    }
                    let r_s =

                     */
                }

                r.narrow(0, 0, state_tensor_results.len() as i64)
            };
        }




        todo!();
    }
    */


    pub fn batch_train_env_rewards(&mut self, trajectories: &[AgentTrajectory<DP, InfoSet>], gamma: f64)
        -> SztormError<DP>
    where for<'a> Tensor: From<&'a <DP as DomainParameters>::UniversalReward>{

        for t in trajectories{
            if t.list().is_empty(){
                continue;
            }
            let state_tensor_results: Vec<Tensor> = t.list().iter().map(|step|{
                self.state_converter.make_tensor(step.step_state())
            }).collect();

            let final_score_t: Tensor =  t.list().last().unwrap().universal_score_after().into();

            let discounted_rewards = {
                let mut r = Tensor::zeros([state_tensor_results.len() as i64 + 1, 1 ], FLOAT_CPU);
                for s in (0..state_tensor_results.len()).rev(){
                    let r_s = Tensor::from(&t[s].step_universal_reward()) + (r.get(s as i64+1) * gamma);
                    r.get(s as i64).copy_(&r_s);
                }

                r.narrow(0,0, state_tensor_results.len() as i64)
            };
        }
        todo!()
    }
}

impl<DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug,
    TB: ConvStateToTensor<InfoSet>,
    /*ActInterpreter: TensorInterpreter<Option<DP::ActionType>>*/
> Policy<DP> for ActorCriticPolicy<DP, InfoSet, TB,
    /*ActInterpreter*/
>
where <DP as DomainParameters>::ActionType: ActionTensor{
    type StateType = InfoSet;

    fn select_action(&self, state: &Self::StateType) -> Option<DP::ActionType> {
        //let state_tensor = self.state_converter.build_tensor(state)
        //    .unwrap_or_else(|_| panic!("Failed converting state to Tensor: {:?}", state));
        let state_tensor = self.state_converter.make_tensor(state);
        let out = tch::no_grad(|| (self.network.net())(&state_tensor));
        let actor = out.actor;
        //somewhen it may be changed with temperature
        let probs = actor.softmax(-1, Float);
        let atensor = probs.multinomial(1, true);
        //self.action_interpreter.interpret_tensor(&atensor)
        Some(DP::ActionType::try_from_tensor(&atensor)
            .expect("Failed converting tensor to action"))

    }
}


impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug,
    TB: ConvStateToTensor<InfoSet>,
    /*ActInterpreter: TensorInterpreter<Option<DP::ActionType>>*/
    > SelfExperiencingPolicy<DP> for ActorCriticPolicy<DP, InfoSet, TB,
    /*ActInterpreter*/>
where DP::ActionType: From<i64>{
    type PolicyUpdateError = tch::TchError;

    fn select_action_and_collect_experience(&mut self) -> Option<DP::ActionType> {
        todo!()
    }


    fn apply_experience(&mut self) -> Result<(), Self::PolicyUpdateError> {
        todo!()
    }
}
