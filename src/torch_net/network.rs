use tch::{TchError, Tensor};
use tch::nn::{Optimizer, OptimizerConfig, Path, VarStore};
use crate::torch_net::{NetOutput, TensorA2C};

pub struct NeuralNet<Output: NetOutput>{
    net: Box<dyn Fn(&Tensor) -> Output + Send>,
    var_store: VarStore,
}


pub type NeuralNet1 = NeuralNet<Tensor>;
pub type NeuralNet2 = NeuralNet<(Tensor, Tensor)>;
pub type A2CNet = NeuralNet<TensorA2C>;

/// To construct network you need `VarStore` and function (closure) taking `nn::Path` as argument
/// and constructs function (closure) which applies network model to `Tensor` producing `NetOutput`,
/// in following example `NetOutput` of `(Tensor, Tensor)` is used for purpose of actor-critic method.
/// # Example:
/// ```
/// use tch::{Device, nn, Tensor};
/// use tch::nn::{Adam, VarStore};
/// use sztorm_rl::torch_net::{A2CNet, NeuralNet2, TensorA2C};
/// let device = Device::cuda_if_available();
/// let var_store = VarStore::new(device);
/// let number_of_actions = 33_i64;
/// let neural_net = A2CNet::new(var_store, |path|{
///     let seq = nn::seq()
///         .add(nn::linear(path / "input", 16, 128, Default::default()))
///         .add(nn::linear(path / "hidden", 128, 128, Default::default()));
///     let actor = nn::linear(path / "al", 128, number_of_actions, Default::default());
///     let critic = nn::linear(path / "cl", 128, 1, Default::default());
///     let device = path.device();
///     {move |xs: &Tensor|{
///         let xs = xs.to_device(device).apply(&seq);
///         //(xs.apply(&critic), xs.apply(&actor))
///         TensorA2C{critic: xs.apply(&critic), actor: xs.apply(&actor)}
///     }}
///
/// });
///
/// let optimizer = neural_net.build_optimizer(Adam::default(), 0.01);
/// ```
impl<Output: NetOutput> NeuralNet<Output>{

    pub fn new<N: 'static + Send + Fn(&Tensor) -> Output,F: Fn(&Path) -> N>(var_store: VarStore, model_closure: F)  -> Self{

        let device = var_store.root().device();
        let model = (model_closure)(&var_store.root());
        Self{
            var_store,
            net: Box::new(move |x| {(model)(&x.to_device(device))})
        }
    }
    pub fn build_optimizer<OptC: OptimizerConfig>
        (&self, optimiser_config: OptC, learning_rate: f64) -> Result<Optimizer, TchError>{

        optimiser_config.build(&self.var_store, learning_rate)
    }
    /// Returns reference to internal network offering `Tensor -> Output` application.
    /// # Example:
    /// ```
    /// use tch::{Device, Kind, nn, Tensor};
    /// use tch::nn::VarStore;
    /// use sztorm_rl::torch_net::NeuralNet;
    /// let device = Device::cuda_if_available();
    /// let var_store = VarStore::new(device);
    /// let neural_net = NeuralNet::new(var_store, |path|{
    ///     let seq = nn::seq()
    ///         .add(nn::linear(path / "input", 32, 4, Default::default()));
    ///     move |tensor|{tensor.apply(&seq)}
    ///
    /// });
    /// let input_tensor = Tensor::zeros(32, (Kind::Float, device));
    /// let output_tensor = (neural_net.net())(&input_tensor);
    /// assert_eq!(output_tensor.size(), vec![4]);
    /// ```
    pub fn net(&self) -> &(dyn Fn(&Tensor) -> Output + Send){&self.net}
}