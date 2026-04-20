import torch
from src.attacks.fgsm import fgsm_attack

def train_with_adversarial(model, train_loader, optimizer, criterion, device, epsilon=0.1, epochs=2):
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            # generate adversarial data
            adv_data = fgsm_attack(model, data, target, epsilon)

            # combine clean + adversarial
            combined_data = torch.cat([data, adv_data])
            combined_target = torch.cat([target, target])

            optimizer.zero_grad()
            output = model(combined_data)
            loss = criterion(output, combined_target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[ADV TRAIN] Epoch {epoch+1}, Loss: {total_loss:.4f}")