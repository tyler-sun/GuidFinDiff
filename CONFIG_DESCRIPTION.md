# Description of .toml config for TabDDPM/GuidFinDiff
First of all, `train.T` and `eval.T` denote preprocessing for training and for evaluation, respectively.  

Here we list non-obvious parameters.  

Main part:
- `seed = 0` -- evaluation seed (and training, but for training it is fixed to 0)
- `parent_dir = "exp/abalone/check"` -- exp folder
- `real_data_path = "data/abalone/"`
- `model_type = "mlp"` -- model type that approximates the reverse process
- `num_numerical_features ` -- a number of numerical features in dataset
- `device = "cuda:0"`

Model params:
- `is_y_cond` -- false for regression, true for classification
- `d_in` -- input dimension (not necessary, since scripts calculate it automatically)
- `num_calsses` -- zero for regression, a number of classes for classification
- `rtdl_params` -- MLP parameters

For guided diffusion synthesis:
- `risk_conditioning` -- determines if model trains as a classifier with y labels or as a regressor with risk values
- `cfg_rate` -- lambda weight or guidance scale for training the conditional and unconditional architectures. can set to over 1 to place high emphasis on the conditional score
- `y_conditional`, `y_cond_weights` -- risk conditional value(s) used for setting the conditional value for generation, where y_conditional is the actual conditional value and y_cond_weights a ratio of minority samples generated at that risk level with respect to 'num_samples' (aka 'num_samples * 'y_cond_weights' at the index of y_conditional). note the number of y_conditional and y_cond_weights values need to be the same.
- `guidance_scale` -- lambda weight for importance of the conditional score in generation. typically set to the same value as cfg_rate for more logical evaluation.

Note that for training and sampling with risk values, num_classes should be set to 0 and is_y_cond should be set to true.

```toml
seed = 0
parent_dir = "exp/credit-card/ddpm_tune_best"
real_data_path = "data/credit-card/"
model_type = "mlp"
num_numerical_features = 30
device = "cuda:0"
risk_conditioning = true

[model_params]
is_y_cond = true
num_classes = 0

[model_params.rtdl_params]
d_layers = [
    512,
    1024,
    1024,
    1024,
    1024,
    1024,
    1024,
    256,
]
dropout = 0.0

[diffusion_params]
num_timesteps = 100
gaussian_loss_type = "mse"
scheduler = "cosine"

[train.main]
steps = 30000
lr = 0.0005306160757129948
weight_decay = 0.0
batch_size = 4096
cfg_rate = 0.0

[train.T]
seed = 0
normalization = "quantile"
num_nan_policy = "__none__"
cat_nan_policy = "__none__"
cat_min_frequency = "__none__"
cat_encoding = "__none__"
y_policy = "default"

[sample]
num_samples = 320
batch_size = 100
append = false
limit_filter = 492
boundary_filter = false
classifier = "catboost"
seed = 0
y_conditional = [
    0.8,
]
y_cond_weights = [
    1,
]
y_range = false
guidance_scale = 0.5

[eval.type]
eval_model = "xgboost"
eval_type = "merged"

[eval.T]
seed = 0
normalization = "__none__"
num_nan_policy = "__none__"
cat_nan_policy = "__none__"
cat_min_frequency = "__none__"
cat_encoding = "__none__"
y_policy = "default"


```