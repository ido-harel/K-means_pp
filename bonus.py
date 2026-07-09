import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

MAX_K = 10
ELBOW_K = 3


def main():
    iris = load_iris()
    data = iris.data  # 150 observations, 4 dimensions

    ks = list(range(1, MAX_K + 1))
    inertias = []
    for k in ks:
        model = KMeans(n_clusters=k, init="k-means++", random_state=0)
        model.fit(data)
        inertias.append(model.inertia_)

    fig, ax = plt.subplots()
    ax.plot(ks, inertias, marker="o", color="tab:blue")
    ax.set_title('Elbow Method for selection of optimal "K" clusters')
    ax.set_xlabel("K")
    ax.set_ylabel("Average Dispersion")
    ax.set_xticks(ks)

    # Highlight the elbow point
    elbow_value = inertias[ELBOW_K - 1]
    x_span = ks[-1] - ks[0]
    y_span = max(inertias) - min(inertias)
    circle = Ellipse(
        (ELBOW_K, elbow_value),
        width=0.07 * x_span,
        height=0.12 * y_span,
        fill=False,
        linestyle="--",
        edgecolor="black",
    )
    ax.add_patch(circle)
    ax.annotate(
        "Elbow Point",
        xy=(ELBOW_K + 0.15, elbow_value + 0.06 * y_span),
        xytext=(ELBOW_K + 1.5, elbow_value + 0.28 * y_span),
        arrowprops=dict(arrowstyle="->", linestyle="--", color="black"),
    )

    fig.savefig("elbow.png")


if __name__ == "__main__":
    main()