use std::fmt::Debug;
use std::marker::PhantomData;
use log::{debug, trace};
use tch::Kind::{Float};
use tch::nn::{Optimizer, VarStore};
use tch::{Kind, kind, Tensor};
use sztorm::agent::{AgentTrajectory, Policy};
use sztorm::error::SztormError;
use sztorm::domain::DomainParameters;
use sztorm::state::agent::{InformationSet, ScoringInformationSet};
use crate::error::SztormRLError;
use crate::experiencing_policy::SelfExperiencingPolicy;
use crate::{LearningNetworkPolicy, TrainConfig};
use crate::tensor_repr::{ActionTensor, ConvertToTensor, FloatTensorReward, WayToTensor};
use crate::torch_net::{A2CNet, TensorA2C};


pub struct ActorCriticPolicy<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ConvertToTensor<InfoSetWay>,
    //StateConverter: ConvStateToTensor<InfoSet>,
    InfoSetWay: WayToTensor
    //ActInterpreter: TensorInterpreter<Option<DP::ActionType>>
> {
    network: A2CNet,
    #[allow(dead_code)]
    optimizer: Optimizer,
    _dp: PhantomData<DP>,
    _is: PhantomData<InfoSet>,
    //state_converter: StateConverter,
    convert_way: InfoSetWay,
    training_config: TrainConfig,
    //action_interpreter: ActInterpreter

}

impl<
    DP: DomainParameters,
    InfoSet: ScoringInformationSet<DP>  + Debug + ConvertToTensor<InfoSetWay>,
    InfoSetWay: WayToTensor,
    //InfoSet: ScoringInformationSet<DP> + Debug,
    //StateConverter: ConvStateToTensor<InfoSet>>
    >
ActorCriticPolicy<
    DP,
    InfoSet,
    InfoSetWay,
    //StateConverter>
    >
where <DP as DomainParameters>::ActionType: ActionTensor{
    pub fn new(network: A2CNet,
               optimizer: Optimizer,
               convert_way: InfoSetWay,
               training_config: TrainConfig
               //state_converter: StateConverter,
               /*action_interpreter: ActInterpreter*/
    ) -> Self{
        Self{
            network, optimizer,
            //state_converter,
            convert_way,
            training_config,
            //action_interpreter,
            _dp: Default::default(), _is: Default::default()
            }
    }




    pub fn batch_train_env_rewards(&mut self, trajectories: &[AgentTrajectory<DP, InfoSet>], gamma: f64)
        -> Result<(), SztormError<DP>>
    where /*for<'a> Tensor: From<&'a <DP as DomainParameters>::UniversalReward>,*/
    <DP as DomainParameters>::UniversalReward: FloatTensorReward {


        let device = self.network.device();
        let capacity_estimate = trajectories.iter().fold(0, |acc, x|{
           acc + x.list().len()
        });
        let tmp_capacity_estimate = trajectories.iter().map(|x|{
            x.list().len()
        }).max().unwrap_or(0);
        let mut state_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut reward_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut action_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut discounted_rewards_tensor_vec: Vec<Tensor> = Vec::with_capacity(tmp_capacity_estimate);
        for t in trajectories{


            if t.list().is_empty(){
                continue;
            }
            let steps_in_trajectory = t.list().len();

            let mut state_tensor_vec_t: Vec<Tensor> = t.list().iter().map(|step|{
                //self.state_converter.make_tensor(step.step_state())
                step.step_state().to_tensor(&self.convert_way)
            }).collect();

            let mut action_tensor_vec_t: Vec<Tensor> = t.list().iter().map(|step|{
                step.taken_action().to_tensor().to_kind(kind::Kind::Int64)
            }).collect();

            let final_score_t: Tensor =  t.list().last().unwrap().universal_score_after().to_tensor();

            discounted_rewards_tensor_vec.clear();
            for _ in 0..=steps_in_trajectory{
                discounted_rewards_tensor_vec.push(Tensor::zeros(DP::UniversalReward::total_size(), (Kind::Float, self.network.device())));
            }
            debug!("Discounted_rewards_tensor_vec len before inserting: {}", discounted_rewards_tensor_vec.len());
            //let mut discounted_rewards_tensor_vec: Vec<Tensor> = vec![Tensor::zeros(DP::UniversalReward::total_size(), (Kind::Float, self.network.device())); steps_in_trajectory+1];
            discounted_rewards_tensor_vec.last_mut().unwrap().copy_(&final_score_t);
            for s in (0..discounted_rewards_tensor_vec.len()-1).rev(){
                //println!("{}", s);
                let r_s = &t[s].step_universal_reward().to_tensor() + (&discounted_rewards_tensor_vec[s+1] * gamma);
                discounted_rewards_tensor_vec[s].copy_(&r_s);
            }
            discounted_rewards_tensor_vec.pop();
            debug!("Discounted rewards_tensor_vec after inserting");

            state_tensor_vec.append(&mut state_tensor_vec_t);
            action_tensor_vec.append(&mut action_tensor_vec_t);
            reward_tensor_vec.append(&mut discounted_rewards_tensor_vec);

        }
        let states_batch = Tensor::stack(&state_tensor_vec[..], 0);
        let results_batch = Tensor::stack(&reward_tensor_vec[..], 0);
        let action_batch = Tensor::stack(&action_tensor_vec[..], 0);
        debug!("Size of states batch: {:?}", states_batch.size());
        debug!("Size of result batch: {:?}", results_batch.size());
        debug!("Size of action batch: {:?}", action_batch.size());
        let TensorA2C{actor, critic} = (self.network.net())(&states_batch);
        let log_probs = actor.log_softmax(-1, Kind::Float);
        let probs = actor.softmax(-1, Float);
        let action_log_probs = {
            let index =  action_batch.to_device(self.network.device());
            debug!("Index: {:?}", index);
            log_probs.gather(1, &index, false)
        };

        debug!("Action log probs size: {:?}", action_log_probs.size());
        debug!("Probs size: {:?}", probs.size());

        let dist_entropy = (-log_probs * probs).sum_dim_intlist(-1, false, Float).mean(Float);
        let advantages = results_batch.to_device(device) - critic;
        let value_loss = (&advantages * &advantages).mean(Float);
        let action_loss = (-advantages.detach() * action_log_probs).mean(Float);
        let loss = value_loss * 0.5 + action_loss - dist_entropy * 0.01;
        self.optimizer.backward_step_clip(&loss, 0.5);

        Ok(())
    }

    pub fn batch_train_state_rewards(&mut self, trajectories: &[AgentTrajectory<DP, InfoSet>], gamma: f64)
        -> Result<(), SztormError<DP>>
    where
    <InfoSet as ScoringInformationSet<DP>>::RewardType: FloatTensorReward{
        let device = self.network.device();
        let capacity_estimate = trajectories.iter().fold(0, |acc, x|{
           acc + x.list().len()
        });
        let tmp_capacity_estimate = trajectories.iter().map(|x|{
            x.list().len()
        }).max().unwrap_or(0);
        let mut state_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut reward_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut action_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut discounted_rewards_tensor_vec: Vec<Tensor> = Vec::with_capacity(tmp_capacity_estimate);
        for t in trajectories{


            if t.list().is_empty(){
                continue;
            }
            let steps_in_trajectory = t.list().len();

            let mut state_tensor_vec_t: Vec<Tensor> = t.list().iter().map(|step|{
                //self.state_converter.make_tensor(step.step_state())
                step.step_state().to_tensor(&self.convert_way)
            }).collect();

            let mut action_tensor_vec_t: Vec<Tensor> = t.list().iter().map(|step|{
                step.taken_action().to_tensor().to_kind(kind::Kind::Int64)
            }).collect();

            let final_score_t: Tensor =  t.list().last().unwrap().subjective_score_after().to_tensor();

            discounted_rewards_tensor_vec.clear();
            for _ in 0..=steps_in_trajectory{
                discounted_rewards_tensor_vec.push(Tensor::zeros(InfoSet::RewardType::total_size(), (Kind::Float, self.network.device())));
            }
            debug!("Discounted_rewards_tensor_vec len before inserting: {}", discounted_rewards_tensor_vec.len());
            //let mut discounted_rewards_tensor_vec: Vec<Tensor> = vec![Tensor::zeros(DP::UniversalReward::total_size(), (Kind::Float, self.network.device())); steps_in_trajectory+1];
            discounted_rewards_tensor_vec.last_mut().unwrap().copy_(&final_score_t);
            for s in (0..discounted_rewards_tensor_vec.len()-1).rev(){
                //println!("{}", s);
                let r_s = &t[s].step_subjective_reward().to_tensor() + (&discounted_rewards_tensor_vec[s+1] * gamma);
                discounted_rewards_tensor_vec[s].copy_(&r_s);
            }
            discounted_rewards_tensor_vec.pop();
            debug!("Discounted rewards_tensor_vec after inserting");

            state_tensor_vec.append(&mut state_tensor_vec_t);
            action_tensor_vec.append(&mut action_tensor_vec_t);
            reward_tensor_vec.append(&mut discounted_rewards_tensor_vec);

        }
        let states_batch = Tensor::stack(&state_tensor_vec[..], 0);
        let results_batch = Tensor::stack(&reward_tensor_vec[..], 0);
        let action_batch = Tensor::stack(&action_tensor_vec[..], 0);
        debug!("Size of states batch: {:?}", states_batch.size());
        debug!("Size of result batch: {:?}", results_batch.size());
        debug!("Size of action batch: {:?}", action_batch.size());
        let TensorA2C{actor, critic} = (self.network.net())(&states_batch);
        let log_probs = actor.log_softmax(-1, Kind::Float);
        let probs = actor.softmax(-1, Float);
        let action_log_probs = {
            let index =  action_batch.to_device(self.network.device());
            debug!("Index: {}", index);
            log_probs.gather(1, &index, false)
        };

        debug!("Action log probs size: {:?}", action_log_probs.size());
        debug!("Probs size: {:?}", probs.size());

        let dist_entropy = (-log_probs * probs).sum_dim_intlist(-1, false, Float).mean(Float);
        let advantages = results_batch.to_device(device) - critic;
        let value_loss = (&advantages * &advantages).mean(Float);
        let action_loss = (-advantages.detach() * action_log_probs).mean(Float);
        let loss = value_loss * 0.5 + action_loss - dist_entropy * 0.01;
        self.optimizer.backward_step_clip(&loss, 0.5);

        Ok(())


    }
}

impl<DP: DomainParameters,
    //InfoSet: InformationSet<DP> + Debug,
    //TB: ConvStateToTensor<InfoSet>,
    InfoSet: InformationSet<DP> + Debug + ConvertToTensor<InfoSetWay>,
    InfoSetWay: WayToTensor,
    /*ActInterpreter: TensorInterpreter<Option<DP::ActionType>>*/
> Policy<DP> for ActorCriticPolicy<
    DP,
    InfoSet,
    //TB,
    InfoSetWay,
    /*ActInterpreter*/
>
where <DP as DomainParameters>::ActionType: ActionTensor{
    type StateType = InfoSet;

    fn select_action(&self, state: &Self::StateType) -> Option<DP::ActionType> {
        //let state_tensor = self.state_converter.build_tensor(state)
        //    .unwrap_or_else(|_| panic!("Failed converting state to Tensor: {:?}", state));
        //let state_tensor = self.state_converter.make_tensor(state);
        trace!("Selecting action");
        let state_tensor = state.to_tensor(&self.convert_way);
        let out = tch::no_grad(|| (self.network.net())(&state_tensor));
        let actor = out.actor;
        //somewhen it may be changed with temperature
        let probs = actor.softmax(-1, Float);
        let atensor = probs.multinomial(1, true);
        trace!("After selecting action, before converting from tensor to action form");
        //self.action_interpreter.interpret_tensor(&atensor)
        Some(DP::ActionType::try_from_tensor(&atensor)
            .expect("Failed converting tensor to action"))

    }
}


impl<
    DP: DomainParameters,
    InfoSet: ScoringInformationSet<DP>  + Debug + ConvertToTensor<InfoSetWay>,
    InfoSetWay: WayToTensor,
    //InfoSet: ScoringInformationSet<DP> + Debug,
    //StateConverter: ConvStateToTensor<InfoSet>>
    > LearningNetworkPolicy<DP> for ActorCriticPolicy<DP, InfoSet, InfoSetWay>
where <DP as DomainParameters>::ActionType: ActionTensor,
<InfoSet as ScoringInformationSet<DP>>::RewardType: FloatTensorReward{
    type Network = A2CNet;
    type TrainConfig = TrainConfig;

    fn network(&self) -> &A2CNet{
        &self.network
    }

    fn network_mut(&mut self) -> &mut A2CNet{
        &mut self.network
    }

    fn var_store(&self) -> &VarStore{
        self.network.var_store()
    }

    fn var_store_mut(&mut self) -> &mut VarStore{
        self.network.var_store_mut()
    }


    fn batch_train_on_universal_rewards(&mut self, trajectories: &[AgentTrajectory<DP, <Self as Policy<DP>>::StateType>]) -> Result<(), SztormRLError<DP>> {
        let device = self.network.device();
        let capacity_estimate = trajectories.iter().fold(0, |acc, x|{
           acc + x.list().len()
        });
        let tmp_capacity_estimate = trajectories.iter().map(|x|{
            x.list().len()
        }).max().unwrap_or(0);
        let mut state_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut reward_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut action_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut discounted_rewards_tensor_vec: Vec<Tensor> = Vec::with_capacity(tmp_capacity_estimate);
        for t in trajectories{


            if t.list().is_empty(){
                continue;
            }
            let steps_in_trajectory = t.list().len();

            let mut state_tensor_vec_t: Vec<Tensor> = t.list().iter().map(|step|{
                //self.state_converter.make_tensor(step.step_state())
                step.step_state().to_tensor(&self.convert_way)
            }).collect();

            let mut action_tensor_vec_t: Vec<Tensor> = t.list().iter().map(|step|{
                step.taken_action().to_tensor().to_kind(kind::Kind::Int64)
            }).collect();

            let final_score_t: Tensor =  t.list().last().unwrap().subjective_score_after().to_tensor();

            discounted_rewards_tensor_vec.clear();
            for _ in 0..=steps_in_trajectory{
                discounted_rewards_tensor_vec.push(Tensor::zeros(InfoSet::RewardType::total_size(), (Kind::Float, self.network.device())));
            }
            debug!("Discounted_rewards_tensor_vec len before inserting: {}", discounted_rewards_tensor_vec.len());
            //let mut discounted_rewards_tensor_vec: Vec<Tensor> = vec![Tensor::zeros(DP::UniversalReward::total_size(), (Kind::Float, self.network.device())); steps_in_trajectory+1];
            discounted_rewards_tensor_vec.last_mut().unwrap().copy_(&final_score_t);
            for s in (0..discounted_rewards_tensor_vec.len()-1).rev(){
                //println!("{}", s);
                let r_s = &t[s].step_subjective_reward().to_tensor().to_device(device) + (&discounted_rewards_tensor_vec[s+1] * self.training_config.gamma);
                discounted_rewards_tensor_vec[s].copy_(&r_s);
            }
            discounted_rewards_tensor_vec.pop();
            debug!("Discounted rewards_tensor_vec after inserting");

            state_tensor_vec.append(&mut state_tensor_vec_t);
            action_tensor_vec.append(&mut action_tensor_vec_t);
            reward_tensor_vec.append(&mut discounted_rewards_tensor_vec);

        }
        let states_batch = Tensor::stack(&state_tensor_vec[..], 0).to_device(device);
        let results_batch = Tensor::stack(&reward_tensor_vec[..], 0).to_device(device);
        let action_batch = Tensor::stack(&action_tensor_vec[..], 0).to_device(device);
        debug!("Size of states batch: {:?}", states_batch.size());
        debug!("Size of result batch: {:?}", results_batch.size());
        debug!("Size of action batch: {:?}", action_batch.size());
        let TensorA2C{actor, critic} = (self.network.net())(&states_batch);
        let log_probs = actor.log_softmax(-1, Kind::Float);
        let probs = actor.softmax(-1, Float);
        let action_log_probs = {
            let index =  action_batch.to_device(self.network.device());
            debug!("Index: {}", index);
            log_probs.gather(1, &index, false)
        };

        debug!("Action log probs size: {:?}", action_log_probs.size());
        debug!("Probs size: {:?}", probs.size());

        let dist_entropy = (-log_probs * probs).sum_dim_intlist(-1, false, Float).mean(Float);
        let advantages = results_batch.to_device(device) - critic;
        let value_loss = (&advantages * &advantages).mean(Float);
        let action_loss = (-advantages.detach() * action_log_probs).mean(Float);
        let loss = value_loss * 0.5 + action_loss - dist_entropy * 0.01;
        self.optimizer.backward_step_clip(&loss, 0.5);

        Ok(())
    }

    fn config(&self) -> &Self::TrainConfig {
        &self.training_config
    }
}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ConvertToTensor<InfoSetWay>,
    InfoSetWay: WayToTensor,
    //InfoSet: InformationSet<DP> + Debug,
    //TB: ConvStateToTensor<InfoSet>,
    /*ActInterpreter: TensorInterpreter<Option<DP::ActionType>>*/
    >
SelfExperiencingPolicy<DP> for ActorCriticPolicy<
    DP,
    InfoSet,
    //TB,
    InfoSetWay,
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


