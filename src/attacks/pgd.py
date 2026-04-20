import torch
import torch.nn.functional as F

def pgd_attack(model, data, target, epsilon=0.3, alpha=0.01, iters=40):
    original_data = data.clone().detach()
    perturbed_data = data.clone().detach()

    for _ in range(iters):
        perturbed_data.requires_grad_(True)

        output = model(perturbed_data)
        loss = F.cross_entropy(output, target)

        model.zero_grad()
        loss.backward()

        grad = perturbed_data.grad

        # gradient step
        perturbed_data = perturbed_data + alpha * grad.sign()

        # project back into epsilon ball
        perturbation = torch.clamp(
            perturbed_data - original_data,
            min=-epsilon,
            max=epsilon
        )

        perturbed_data = torch.clamp(original_data + perturbation, 0, 1).detach()

    return perturbed_data