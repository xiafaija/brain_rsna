import torch
from dataset import train_loader, valid_loader
from model import Model, LossMeter, AccMeter
from torch.nn import functional as torch_functional
from train import Trainer
from dataset import train_data_retriever, valid_data_retriever


train_num = len(train_data_retriever)
val_num = len(valid_data_retriever)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = train_loader()
valid_loader = valid_loader()
model = Model()
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch_functional.binary_cross_entropy_with_logits

trainer = Trainer(
    model,
    device,
    optimizer,
    criterion,
    LossMeter,
    AccMeter
)

history = trainer.fit(
    2,
    train_loader,
    valid_loader,
    f"best-model-0.pth",
    2,
    train_num,
    val_num
)
