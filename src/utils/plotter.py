import matplotlib.pyplot as plt

def plot_accuracy(results, save_path):
    defenses = list(results.keys())

    clean = [results[d]["clean"] for d in defenses]
    plt.plot(defenses, clean, marker='o', label="Clean")

    fgsm = [results[d]["fgsm"] for d in defenses]
    pgd = [results[d]["pgd"] for d in defenses]

    plt.figure()
    plt.plot(defenses, fgsm, marker='o', label="FGSM")
    plt.plot(defenses, pgd, marker='o', label="PGD")

    plt.xlabel("Defense")
    plt.ylabel("Accuracy")
    plt.title("Attack vs Defense Accuracy")
    plt.legend()

    plt.savefig(f"{save_path}/accuracy_plot.png")
    plt.close()


def plot_deepfool(results, save_path):
    defenses = list(results.keys())
    values = [results[d]["deepfool"] for d in defenses]

    plt.figure()
    plt.bar(defenses, values)

    plt.xlabel("Defense")
    plt.ylabel("Perturbation (L2)")
    plt.title("DeepFool Perturbation")

    plt.savefig(f"{save_path}/deepfool_plot.png")
    plt.close()