import torch

def feature_squeezing(x, bits=5):
    levels = 2 ** bits
    return torch.round(x * levels) / levels
    
def train_with_preprocessing_only(model, loader, optimizer, criterion, device, epochs=3):
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for data, target in loader:
            data, target = data.to(device), target.to(device)

            data = feature_squeezing(data)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[PREPROCESS] Epoch {epoch+1} | Loss: {total_loss:.4f}")