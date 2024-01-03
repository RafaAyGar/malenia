import matplotlib.pyplot as plt


def plot_Boxplot(results, metric_name, figsize=(16, 8)):
    plt.style.use("seaborn")
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["figure.autolayout"] = True

    title_font = {"color": "black", "weight": "bold", "verticalalignment": "bottom"}

    metric_results_by_dataset = results.get_results_by_dataset_metric(metric_name)
    metric_results_by_dataset.plot(
        kind="box", title="Comparison in terms of " + metric_name + " metric"
    )
    plt.title("Comparison in terms of " + metric_name, **title_font)
    plt.ylim(0.0, metric_results_by_dataset.to_numpy().max() + 0.5)
    plt.show()
