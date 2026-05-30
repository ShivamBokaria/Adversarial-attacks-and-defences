import torch
import torch.nn.functional as F

def pgd_attack(model, x, y, epsilon, alpha, iters):

    ori = x.detach()

    adv = ori + torch.randn_like(ori) * 0.001
    adv = torch.clamp(adv, ori.min(), ori.max())

    for _ in range(iters):
        adv.requires_grad = True

        out = model(adv)
        loss = F.cross_entropy(out, y)

        model.zero_grad()
        loss.backward()

        adv = adv + alpha * adv.grad.sign()

        eta = torch.clamp(adv - ori, -epsilon, epsilon)
        adv = torch.clamp(ori + eta, ori.min(), ori.max()).detach()

    return adv