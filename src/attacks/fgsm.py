import torch
import torch.nn.functional as F

def fgsm_attack(model, data, target, epsilon=0.05):
    data = data.clone().detach().requires_grad_(True)

    output = model(data)
    loss = F.cross_entropy(output, target)

    model.zero_grad()
    loss.backward()

    data_grad = data.grad.data

    perturbed_data = data + epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)

    return perturbed_data