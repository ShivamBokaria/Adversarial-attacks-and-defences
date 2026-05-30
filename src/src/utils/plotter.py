import matplotlib.pyplot as plt

def plot_accuracy(results, path):
    names = list(results.keys())
    clean = [results[k]["clean"] for k in names]

    plt.figure()
    plt.bar(names, clean)
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy")
    plt.savefig(f"{path}/accuracy.png")


def plot_deepfool(results, path):
    names = list(results.keys())
    df = [results[k]["deepfool"] for k in names]

    plt.figure()
    plt.bar(names, df)
    plt.xticks(rotation=45)
    plt.ylabel("DeepFool")
    plt.savefig(f"{path}/deepfool.png")