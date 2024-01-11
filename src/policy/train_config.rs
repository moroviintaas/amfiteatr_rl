use crate::policy::DiscountAggregator;

#[derive(Copy, Clone)]
pub struct TrainConfig{
    pub gamma: f64
}

impl DiscountAggregator for TrainConfig{
    fn discount_factor(&self) -> f64 {
        self.gamma
    }
}