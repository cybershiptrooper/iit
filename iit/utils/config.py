import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WANDB_ENTITY = "cybershiptrooper" #TODO: This should be editable by the user at runtime