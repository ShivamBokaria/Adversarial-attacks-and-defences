import torch
import torch.nn.functional as F

def fgsm_attack(model, x, y, epsilon=0.1):
    x = x.clone().detach().requires_grad_(True)

    output = model(x)
    loss = F.cross_entropy(output, y)

    model.zero_grad()
    loss.backward()

    adv = x + epsilon * x.grad.sign()
    adv = torch.clamp(adv, 0, 1)

    return adv.detach()