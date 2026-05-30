from src.attacks.pgd import pgd_attack

def train_with_adversarial(model, loader, optimizer, criterion, device, dataset, epochs):
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if dataset == "mnist":
                adv = pgd_attack(model, x, y, 0.3, 0.01, 8)
            else:
                adv = pgd_attack(model, x, y, 0.03, 0.007, 8)

            optimizer.zero_grad()
            loss = criterion(model(adv), y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[ADV] Epoch {epoch+1} | Loss: {total_loss:.2f}")