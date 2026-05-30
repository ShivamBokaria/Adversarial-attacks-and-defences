import torch
import foolbox as fb

def deepfool_attack(model, x, y, device):
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    attack = fb.attacks.L2DeepFoolAttack()

    raw, clipped, success = attack(fmodel, x, y, epsilons=None)

    return clipped