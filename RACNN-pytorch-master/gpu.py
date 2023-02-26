import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

x = torch.ones(1, 10)
print(x.to(device))
