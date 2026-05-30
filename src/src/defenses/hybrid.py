import torch
from src.attacks.pgd import pgd_attack
from src.defenses.preprocessing import feature_squeezing

def train_with_hybrid(model, loader, optimizer, criterion, device, dataset, epochs):
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            x_clean = feature_squeezing(x, bits=5)

            if dataset == "mnist":
                x_adv = pgd_attack(model, x_clean, y, 0.3, 0.01, 8)
            else:
                x_adv = pgd_attack(model, x_clean, y, 0.03, 0.007, 8)

            out_clean = model(x_clean)
            out_adv = model(x_adv)

            # 🔥 final balanced loss
            loss = 0.3 * criterion(out_clean, y) + 0.7 * criterion(out_adv, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[HYBRID] Epoch {epoch+1} | Loss: {total_loss:.2f}")