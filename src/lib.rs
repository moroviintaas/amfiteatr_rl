pub mod torch_net;
pub mod actor_critic;
pub mod experiencing_policy;
pub mod tensor_repr;
pub mod error;
pub mod q_learning_policy;
mod learning_policy;
pub mod agent;
mod train_config;
pub mod demo;

pub use learning_policy::*;
pub use train_config::*;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}



