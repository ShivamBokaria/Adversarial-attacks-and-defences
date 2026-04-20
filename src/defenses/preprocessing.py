import torch

def feature_squeezing(x):
    noise = torch.randn_like(x) * 0.02
    x = x + noise
    x = torch.clamp(x, 0, 1)
    return torch.round(x * 15) / 15

def train_with_preprocessing_only(model, train_loader, optimizer, criterion, device, epochs=3):
    model.train()

    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            data = feature_squeezing(data)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

        print(f"[PREPROCESS ONLY] Epoch {epoch+1}")