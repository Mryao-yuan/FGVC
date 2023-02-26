import torch

print(torch.device('cuda:3' if torch.cuda.is_available() else 'gpu'))
