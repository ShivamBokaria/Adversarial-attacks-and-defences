import torch
import torch.nn as nn
import torch.optim as optim
import time
import warnings
warnings.filterwarnings("ignore")
torch.set_num_threads(4)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.utils.results_manager import get_next_run_folder
from src.utils.plotter import plot_accuracy, plot_deepfool
from src.utils.table_saver import save_results_table

from src.models.cnn import CNN
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.defenses.adv_training import train_with_adversarial
from src.defenses.hybrid import train_with_hybrid
from src.defenses.preprocessing import train_with_preprocessing_only
from src.attacks.deepfool import deepfool_attack

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# -------------------------
# DATA LOADER
# -------------------------
def get_data(dataset="mnist", debug=False):

    if dataset == "mnist":
        transform = transforms.ToTensor()
        train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    elif dataset == "cifar":
        transform = transforms.Compose([transforms.ToTensor()])

        train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

    # 🔴 LIMIT DATA FOR CPU
        train_dataset.data = train_dataset.data[:20000]
        train_dataset.targets = train_dataset.targets[:20000]

        test_dataset.data = test_dataset.data[:5000]
        test_dataset.targets = test_dataset.targets[:5000]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # 🔴 DEBUG MODE (FAST RUN)
    if debug:
        train_loader = list(train_loader)[:200]
        test_loader = list(test_loader)[:100]

    return train_loader, test_loader


# -------------------------
# EVALUATION FUNCTIONS
# -------------------------
def evaluate(model, loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    return correct / len(loader)


def evaluate_fgsm(model, loader, epsilon=0.1):
    correct = 0

    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        adv_data = fgsm_attack(model, data, target, epsilon)

        output = model(adv_data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

    return correct / len(loader)


def evaluate_pgd(model, loader, epsilon=0.3, alpha=0.007, iters=10):
    model.eval()
    correct = 0

    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        adv_data = pgd_attack(model, data, target, epsilon, alpha, iters)

        output = model(adv_data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

    return correct / len(loader)


def evaluate_deepfool(model, loader):
    model.eval()

    total_perturbation = 0
    total_samples = 0

    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        adv_data = deepfool_attack(model, data, target, DEVICE)

        perturbation = (adv_data - data).view(data.size(0), -1)
        norm = torch.norm(perturbation, p=2, dim=1)

        total_perturbation += norm.sum().item()
        total_samples += data.size(0)

    return total_perturbation / total_samples


# -------------------------
# MAIN RUN FUNCTION
# -------------------------
def run(defense="none", model_type="cnn", dataset="mnist", debug=False):

    start_time = time.time()
    print(f"\nStart Time: {time.ctime(start_time)}")

    train_loader, test_loader = get_data(dataset, debug)

    # -------- MODEL --------
    if model_type == "cnn":
        model = CNN().to(DEVICE)

    elif model_type == "resnet":
        from src.models.resnet import ResNet18

        if dataset == "mnist":
            raise ValueError("ResNet requires CIFAR")

        model = ResNet18().to(DEVICE)

    # -------- OPTIM --------
    lr = 0.001 if dataset == "mnist" else 0.0005
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 🔴 REDUCED EPOCHS FOR CIFAR
    epochs = 3 if dataset == "mnist" else 5

    print(f"\nRunning {model_type.upper()} on {dataset.upper()} with defense: {defense}\n")

    # -------- TRAIN --------
    if defense == "adv":
        train_with_adversarial(model, train_loader, optimizer, criterion, DEVICE)

    elif defense == "preprocess":
        train_with_preprocessing_only(model, train_loader, optimizer, criterion, DEVICE)

    elif defense == "hybrid":
        train_with_hybrid(model, train_loader, optimizer, criterion, DEVICE)

    else:
        model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                # 🔴 PROGRESS PRINT
                if batch_idx % 20 == 0:
                    print(f"[Epoch {epoch+1}] Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

            print(f"[NORMAL TRAIN] Epoch {epoch+1}")

    # -------- EVALUATE --------
    clean_acc = evaluate(model, test_loader)
    fgsm_acc = evaluate_fgsm(model, test_loader)
    pgd_acc = evaluate_pgd(model, test_loader)
    deepfool_score = evaluate_deepfool(model, test_loader)

    end_time = time.time()

    print("\nRESULTS:")
    print(f"Clean: {clean_acc:.4f}")
    print(f"FGSM: {fgsm_acc:.4f}")
    print(f"PGD: {pgd_acc:.4f}")
    print(f"DeepFool: {deepfool_score:.4f}")

    print(f"\nEnd Time: {time.ctime(end_time)}")
    print(f"Total Time: {(end_time - start_time):.2f} seconds")

    return {
        "clean": clean_acc,
        "fgsm": fgsm_acc,
        "pgd": pgd_acc,
        "deepfool": deepfool_score,
        "time": end_time - start_time
    }


# -------------------------
# MAIN EXECUTION
# -------------------------
if __name__ == "__main__":

    run_path, graphs_path, tables_path = get_next_run_folder()

    results = {}

    # # CNN + MNIST
    # results["cnn_baseline"] = run("none", "cnn", "mnist")
    # results["cnn_preprocess"] = run("preprocess", "cnn", "mnist")  # ✅ ADD
    # results["cnn_adv"] = run("adv", "cnn", "mnist")
    # results["cnn_hybrid"] = run("hybrid", "cnn", "mnist")


    #RESNET + CIFAR
    results["resnet_baseline"] = run("none", "resnet", "cifar")
    results["resnet_preprocess"] = run("preprocess", "resnet", "cifar")
    results["resnet_adv"] = run("adv", "resnet", "cifar")
    results["resnet_hybrid"] = run("hybrid", "resnet", "cifar")

    plot_accuracy(results, graphs_path)
    plot_deepfool(results, graphs_path)
    save_results_table(results, tables_path)

    print(f"\nResults saved in: {run_path}")