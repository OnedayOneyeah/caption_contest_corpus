import torch

a = torch.rand(2, 5)
# b = torch.Longtensor(torch.ones(2))
b = torch.ones(2)
print(a, b)
idx = torch.arange(len(a))
print(a[idx, b.tolist()])