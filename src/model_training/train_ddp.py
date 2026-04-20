import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

def setup(rank, world_size):
    # FIXED: Proper dict assignment, no more TypeError
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class QueryDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Ensure numeric labels
        self.data['label'] = pd.factorize(self.data['ground_truth_segmentation'])[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['raw_query'])
        label = self.data.iloc[idx]['label']
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train(rank, world_size, data_csv):
    setup(rank, world_size)

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Match the 4 labels from your prompt
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4).to(rank)
    model = DDP(model, device_ids=[rank])

    dataset = QueryDataset(data_csv, tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    if rank == 0:
        print(f"--- Starting DDP Training on {world_size} GPUs ---")

    for epoch in range(3):
        sampler.set_epoch(epoch)
        model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'].to(rank),
                attention_mask=batch['attention_mask'].to(rank),
                labels=batch['labels'].to(rank)
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch+1} complete. Loss: {loss.item():.4f}")

    if rank == 0:
        save_path = "/home/amanr.mds2024/Developer/AML/GenAI_proj/models/distilled_query_model"
        model.module.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model successfully saved to {save_path}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, "/home/amanr.mds2024/Developer/AML/GenAI_proj/data/gold_dataset/training_data.csv"), nprocs=world_size, join=True)
