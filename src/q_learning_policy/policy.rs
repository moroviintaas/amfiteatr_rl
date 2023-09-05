use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, Sub};
use tch::Kind::Float;
use tch::nn::Optimizer;
use tch::Tensor;
use sztorm::agent::Policy;
use sztorm::protocol::DomainParameters;
use sztorm::{ProportionalReward, Reward};
use sztorm::state::agent::InformationSet;
use crate::tensor_repr::{ConvertToTensor, WayToTensor};
use crate::torch_net::NeuralNet1;



#[derive(Debug, Copy, Clone)]
pub enum QSelector{
    Max,
    //MultinomialLinear,
    MultinomialLogits
}


impl QSelector{

    pub fn select_q_value_index(&self, q_vals: &Tensor) -> Option<usize>{
        match self{
            Self::Max => {
                let rv = Vec::<f32>::try_from(q_vals.argmax(None, false));
                //rv.map(|v| v.first()).ok().and_then(|i| Some(i as usize))
                rv.ok().and_then(|v|v.first().and_then(|i| Some(*i as usize)))

            },
            Self::MultinomialLogits => {
                let probs = q_vals.softmax(-1, Float);
                let index_t = probs.multinomial(1, false);
                let rv =  Vec::<f32>::try_from(index_t);
                //rv.map(|v|v.first()).ok().and_then(|i| Some(i as usize))
                rv.ok().and_then(|v|v.first().and_then(|i| Some(*i as usize)))
            }
        }
    }

    /*
    fn logits<R: ProportionalReward<f32>>(values: &[R]) -> Vec<f32>
    where for<'a> &'a R: Sub<&'a R, Output = R>
    {
        let sum = values.iter().fold(R::neutral(), |acc, x|{
              acc + x
        });
        values.iter().map(|v|{
            v.proportion(&(&sum - v)).ln()
        }).collect()


    }*/

    /*
    pub fn select_max<R: Reward + Ord>(&self, values: &[R]) -> Option<usize>{
        values.iter().enumerate().max_by(|(_, a), (_, b)| a.cmp(b)).map(|(i,_)| i)
    }
    //pub fn select_linear<>:

    pub fn select_index<R: ProportionalReward<f32> + Ord>(&self, values: &[R] ) -> Option<usize>{
        match self{
            QSelector::Max => self.select_max(values),
            QSelector::MultinomialLinear => {
                todo!()
            }
            QSelector::MultinomialLogits => {
                todo!()
            }
        }
    }

     */
}

pub struct QLearningPolicy<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ConvertToTensor<W2T>,
    W2T: WayToTensor,

>{

    network: NeuralNet1,
    optimizer: Optimizer,
    _dp: PhantomData<DP>,
    _is: PhantomData<InfoSet>,
    convert_way: W2T
}

impl
<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ConvertToTensor<W2T>,
    W2T: WayToTensor
> QLearningPolicy<DP, InfoSet, W2T> {

    pub fn new(network: NeuralNet1, optimizer: Optimizer, convert_way: W2T) -> Self{
        Self{network, optimizer, convert_way, _dp: Default::default(), _is: Default::default()}
    }



}


impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ConvertToTensor<IS2T>,
    IS2T: WayToTensor
> Policy<DP> for QLearningPolicy<DP, InfoSet, IS2T>{
    type StateType = InfoSet;

    fn select_action(&self, state: &Self::StateType) -> Option<DP::ActionType> {

        /*
        let q_predictions : Vec<f32> = state.available_actions().into_iter().map(|a|{

        })

         */
        todo!();
    }
}