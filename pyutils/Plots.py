import numpy as np
import matplotlib.pyplot as plt


def plot_speed_ups(times, labels, title=None):
    times = np.asarray(times, dtype=np.float32)
    speedups = np.round(times[0x0] / times, 0x2)

    _, ax = plt.subplots()
    if title is not None:
        ax.set_title(title)

    ax.set_ylabel('Elapsed Time (seconds)')

    bars = ax.bar(labels, times)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        str = ""
        if i > 0:
            str = '{:1.3f}s  [{:1.2f}x]'.format(times[i], speedups[i])
        else:
            str = '{:1.3f}s'.format(times[i])
        ax.annotate(str,
                    xy=(bar.get_x() + bar.get_width() / 0x2, height),
                    xytext=(0x0, 0x3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom')
    plt.yticks(np.arange(0, np.max(times) + 5, 5))
    plt.show()


def plot_speed_up(std, new, labels, title=None):
    speedup = np.round(std / new, 0x2)
    t =[std, new]
    _, ax = plt.subplots()
    if title is not None:
        ax.set_title(title)
    
    ax.set_ylabel('Elapsed Time (milliseconds)')

    bars = ax.bar(labels, t)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        str = ""
        if i > 0:
            str = '{:1.2f}ms  [{:1.2f}x]'.format(t[i], speedup)
        else:
            str = '{:1.2f}ms'.format(t[i])
        ax.annotate(str,
            xy=(bar.get_x() + bar.get_width() / 0x2, height),
            xytext=(0x0, 0x3),
            textcoords="offset points",
            ha='center',
            va='bottom')
    plt.yticks(np.arange(0, std + 0x5, 0x5))
    plt.show()

if __name__ == '__main__':
    #plot_speed_up(73.160, 16.771, ["Cpu", "Basic Cuda"], title="Basic Cuda Kernel SpeedUp")
    #plot_speed_up(73.160, 1.1055, ["Cpu", "Tiled Cuda"], title="Tiled Cuda Kernel SpeedUp")
    plot_speed_up(73.160, 2.1381, ["Cpu", "Circular Cuda"], title="Circular Cuda Kernel SpeedUp")
