import torch
from tqdm import tqdm
import json
import numpy as np
from iit.tasks.ioi.ioi_hl import IOI_HL
from functools import partial

class IOI_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, hl_model, device):
        self.hl_model = hl_model
        self.data = self.make_data(data, tokenizer)
        self.device = device
        self.tokenizer = tokenizer
    
    def make_data(self, data, tokenizer) -> list[tuple[torch.Tensor, torch.Tensor]]:
        dataset = []
        for example in tqdm(data, desc="Making dataset"):
            input_toks = torch.tensor(tokenizer.encode(example['text']))
            counterfactuals = example['counterfactuals']
            for c_type, counterfactual in counterfactuals.items():
                if 'abc' in c_type:
                    continue
                c_toks = torch.tensor(tokenizer.encode(counterfactual))
                dataset.append((input_toks, c_toks))
        return dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        i, c = self.data[idx]
        i_label = self.hl_model(i).argmax(dim=-1)[-1]
        c_label = self.hl_model(c).argmax(dim=-1)[-1]
        i_label = torch.cat((i[1:], i_label.view(1)))
        c_label = torch.cat((c[1:], c_label.view(1)))
        return (i, i_label), (c, c_label)

    def get_encoded_input_from_torch_input(self, batch, device):
        zipped_data = tuple(zip(*batch))
        x_in, y_in = zipped_data[0:2]
        pad_tok = self.tokenizer.eos_token_id
        max_len = max([len(x) for x in x_in])
        # left pad
        x_in = [torch.cat((torch.tensor([pad_tok] * (max_len - len(x))), x)) for x in x_in]
        y_in = [torch.cat((torch.tensor([pad_tok] * (max_len - len(x))), x)) for x in y_in]
        x_in = torch.stack(x_in).to(device).long()
        y_in = torch.stack(y_in).to(device).long()
        return x_in, y_in
    
    def collate_fn(self, batch, device):
        if not isinstance(batch, list):
            # if batch is a single element, because batch_size was 1 or None, it is a tuple instead of a list
            batch_list = [batch]
        else:
            batch_list = batch
        
        base_input_list, ablation_input_list = zip(*batch_list)
        return self.get_encoded_input_from_torch_input(
            base_input_list, device
        ), self.get_encoded_input_from_torch_input(ablation_input_list, device)
    
    def make_loader(self, batch_size, num_workers):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=partial(self.collate_fn, device=self.device)
        )


def make_ioi_dataset_and_hl(tokenizer, device, json_file="train", train_size=0.8, seed=42):
    data = json.load(open(f"ioi_data/{json_file}.json"))
    train_data, test_data = torch.utils.data.random_split(
        data, 
        [train_size, 1 - train_size], 
        generator=torch.Generator().manual_seed(seed)
    )

    names = json.load(open(f"ioi_data/names/names_{json_file}.json"))
    hl_model = IOI_HL(
        d_vocab=len(tokenizer.vocab),
        names=torch.tensor(np.ravel([tokenizer.encode(" " + name) for name in names]))
    )
    train_dataset = IOI_Dataset(train_data, tokenizer, hl_model, device)
    test_dataset = IOI_Dataset(test_data, tokenizer, hl_model, device)
    return train_dataset, test_dataset, hl_model
