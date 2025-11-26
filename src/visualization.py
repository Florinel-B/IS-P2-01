import matplotlib.pyplot as plt


def plot_incidencias(df):
    plt.figure()
    plt.scatter(df["tiempo"], df["incidencia"])
    plt.title("Incidencias detectadas")
    plt.xlabel("Tiempo")
    plt.ylabel("Tipo incidencia")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
