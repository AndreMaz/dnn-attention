from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


def myFun(a, b):
    print('Received', a)
    return a+b


if __name__ == "__main__":
    myFun(2, 2)
