from src.defenses.preprocessing import feature_squeezing
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
import torch



def train_with_hybrid(model, train_loader, optimizer, criterion, device, epsilon=0.1, epochs=2):
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            # Step 1: preprocess (defense layer 1)
            data = feature_squeezing(data)

            # Step 2: generate adversarial samples
            adv_data = pgd_attack(
                model,
                data,
                target,
                epsilon=0.1,
                alpha=0.01,
                iters=10
            )
            
            # Step 3: combine
            combined_data = torch.cat([data, adv_data])
            combined_target = torch.cat([target, target])

            optimizer.zero_grad()
            output = model(combined_data)
            loss = criterion(output, combined_target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[HYBRID TRAIN] Epoch {epoch+1}, Loss: {total_loss:.4f}")