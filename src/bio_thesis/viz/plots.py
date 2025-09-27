
import matplotlib.pyplot as plt
def save_bar(series, path, title, xlabel, ylabel):
    plt.figure()
    series.plot(kind='bar')
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(path); plt.close()
