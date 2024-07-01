from iit.model_pairs.iit_behavior_model_pair import IITBehaviorModelPair
from iit.tasks.task_loader import *

training_args = {
    "lr": 0.001,
    "early_stop": True,
}
dataset_config = {
    "train_size": 60_000,
    "test_size": 10_000,
    "batch_size": 256,
    "num_workers": 0,
}
task = "mnist_pvr"
train_set, test_set = get_dataset(task, dataset_config=dataset_config)
ll_model, hl_model, corr = get_alignment(
    task, config={"input_shape": test_set.base_data.get_input_shape()}
)
model_pair = IITBehaviorModelPair(
    ll_model=ll_model, hl_model=hl_model, corr=corr, training_args=training_args
)  # TODO: add wrapper for choosing model pair
model_pair.train(train_set, test_set, epochs=10, use_wandb=False)

print(f"done training")
