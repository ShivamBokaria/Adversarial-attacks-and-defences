import foolbox as fb
import torch

def deepfool_attack(model, data, target, device):
    model.eval()

    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    attack = fb.attacks.LinfDeepFoolAttack()

    raw, clipped, success = attack(
        fmodel,
        data,
        target,
        epsilons=None  
    )

    # Foolbox returns tensor directly when epsilons=None
    return clipped.detach()