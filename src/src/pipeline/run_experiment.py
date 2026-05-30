import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.models.cnn import CNN
from src.models.resnet import ResNet18

from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.attacks.deepfool import deepfool_attack

from src.defenses.preprocessing import feature_squeezing
from src.defenses.adv_training import train_with_adversarial
from src.defenses.hybrid import train_with_hybrid

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# RUN FOLDER
# -------------------------
def get_run_folder():
    base = "results"
    os.makedirs(base, exist_ok=True)

    existing = [int(x.split("_")[1]) for x in os.listdir(base) if x.startswith("run_")]
    run_id = max(existing) + 1 if existing else 1

    path = os.path.join(base, f"run_{run_id}")
    os.makedirs(path)

    return path


# -------------------------
# DATA
# -------------------------
def get_data(dataset):

    if dataset == "mnist":
        transform = transforms.ToTensor()

    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    if dataset == "mnist":
        train = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test = datasets.MNIST("./data", train=False, download=True, transform=transform)
    else:
        train = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        test = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

    return DataLoader(train, 32, shuffle=True), DataLoader(test, 32)


# -------------------------
# EVAL
# -------------------------
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


def evaluate_attack(model, loader, dataset, attack):
    correct, total = 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        if attack == "fgsm":
            eps = 0.1 if dataset == "mnist" else 0.03
            adv = fgsm_attack(model, x, y, eps)

        elif attack == "pgd":
            if dataset == "mnist":
                adv = pgd_attack(model, x, y, 0.3, 0.01, 20)
            else:
                adv = pgd_attack(model, x, y, 0.03, 0.007, 30)

        pred = model(adv).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total


def evaluate_deepfool(model, loader, dataset):
    total_norm, total = 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        # 🔥 FIX: de-normalize for CIFAR
        if dataset == "cifar":
            x_unnorm = (x * 0.5) + 0.5   # [-1,1] → [0,1]
        else:
            x_unnorm = x

        adv = deepfool_attack(model, x_unnorm, y, DEVICE)

        # 🔥 re-normalize back before passing to model
        if dataset == "cifar":
            adv = (adv - 0.5) / 0.5

        diff = (adv - x).view(x.size(0), -1)
        norm = torch.norm(diff, p=2, dim=1)

        total_norm += norm.sum().item()
        total += x.size(0)

    return total_norm / total


# -------------------------
# TRAIN
# -------------------------
def train(model, loader, optimizer, criterion, defense, dataset, epochs):

    if defense == "adv":
        train_with_adversarial(model, loader, optimizer, criterion, DEVICE, dataset, epochs)

    elif defense == "hybrid":
        train_with_hybrid(model, loader, optimizer, criterion, DEVICE, dataset, epochs)

    elif defense == "preprocess":
        model.train()
        for epoch in range(epochs):
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                x = feature_squeezing(x, bits=5)

                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

            print(f"[PREPROCESS] Epoch {epoch+1}")

    else:
        model.train()
        for epoch in range(epochs):
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

            print(f"[NORMAL] Epoch {epoch+1}")


# -------------------------
# PLOT
# -------------------------
def plot_results(results, path, name):
    labels = list(results.keys())

    clean = [results[d]["clean"] for d in labels]
    pgd = [results[d]["pgd"] for d in labels]

    plt.figure()
    plt.plot(labels, clean, marker='o', label="Clean")
    plt.plot(labels, pgd, marker='o', label="PGD")
    plt.legend()
    plt.savefig(os.path.join(path, f"{name}.png"))
    plt.close()


# -------------------------
# RUN
# -------------------------
def run(model_type, dataset, run_path):

    # defenses = ["none", "preprocess", "adv", "hybrid"]
    defenses = [ "adv", "hybrid"]
    results = {}

    epochs = 3 if dataset == "mnist" else 10
    lr = 0.001 if dataset == "mnist" else 0.0005

    for defense in defenses:

        train_loader, test_loader = get_data(dataset)

        model = CNN().to(DEVICE) if model_type == "cnn" else ResNet18().to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        print("\n==============================")
        print(f"{model_type.upper()} | {dataset.upper()} | {defense.upper()}")
        print("==============================")

        train(model, train_loader, optimizer, criterion, defense, dataset, epochs)

        clean = evaluate(model, test_loader)
        fgsm = evaluate_attack(model, test_loader, dataset, "fgsm")
        pgd = evaluate_attack(model, test_loader, dataset, "pgd")
        deep = evaluate_deepfool(model, test_loader, dataset)
        
        results[defense] = {
            "clean": clean,
            "fgsm": fgsm,
            "pgd": pgd,
            "deepfool": deep
        }

        print("------------------------------")
        print(f"Clean     : {clean:.4f}")
        print(f"FGSM      : {fgsm:.4f}")
        print(f"PGD       : {pgd:.4f}")
        print(f"DeepFool  : {deep:.4f}")
        print("------------------------------")

    with open(os.path.join(run_path, f"{model_type}_{dataset}.json"), "w") as f:
        json.dump(results, f, indent=4)

    plot_results(results, run_path, f"{model_type}_{dataset}")


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    print("Using:", DEVICE)

    run_path = get_run_folder()

    # run("cnn", "mnist", run_path)
    run("resnet", "cifar", run_path)

    print(f"\nSaved in: {run_path}")