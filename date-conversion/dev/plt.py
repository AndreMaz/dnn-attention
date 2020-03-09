import matplotlib.pyplot as plt
import tensorflow as tf

def plotter():
    data = tf.random.uniform((10, 10)).numpy()
    print(data)
    plt.matshow(data)
    
    plt.ylabel('input word')
    plt.xlabel('generated word')

    plt.show(block=True)
    return 1

if __name__ == "__main__":
    plotter()