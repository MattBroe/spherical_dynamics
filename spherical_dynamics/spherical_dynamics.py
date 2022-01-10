import math_utils as mu
import numpy as np


def main():
    breakpoint()
    x = np.float32(0)

    while True:
        output = mu.bump(x)
        print("Input: {0} Output: {1}", x, output)
        x = .5 + (x / 2)
        breakpoint()


if __name__ == "__main__":
    main()
