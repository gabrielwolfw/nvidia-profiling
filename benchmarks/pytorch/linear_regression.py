import torch

device = torch.device("cuda")

x = torch.randn(10000, 1024, device=device)
y = torch.randn(10000, 1, device=device)

model = torch.nn.Linear(1024, 1).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for i in range(10000):
    optimizer.zero_grad()

    prediction = model(x)
    loss = ((prediction - y) ** 2).mean()

    loss.backward()
    optimizer.step()

torch.cuda.synchronize()