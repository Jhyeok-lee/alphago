import numpy as np

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

a = [-1, 0, 1]
print(softmax(a))

def s(x):
	x[1] = 1
	print(x)

c = list(a)
s(a)
print(a)
print(c)