def next_height(x, i):
    if i == 0:
        print(x)
        return x
    else:
        print(0.516*x + 0.8567)
        return next_height(0.516*x + 0.8567, i - 1)

next_height(2.2, 20)
