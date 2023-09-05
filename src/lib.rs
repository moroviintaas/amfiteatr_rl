pub mod torch_net;
pub mod actor_critic;
pub mod experiencing_policy;
pub mod tensor_repr;
pub mod error;
pub mod q_learning_policy;
mod learning_policy;
pub use learning_policy::*;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

