from MultiLayerPerceptron import MLP
from ValueClass import Value

def calulateBackwardPropagation(xs, ys, n):
    ypred = list()
    for k in range(100):
        ypred = [n(x) for x in xs]
        loss = sum((yact - yp) ** 2 for yact, yp in zip(ys, ypred))

        loss.backward()

        for p in n.parameters():
            p.data += -0.01 * p.grad
            p.grad = 0.0

        print(k, loss.data)

if __name__ == '__main__':
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = [1.0, -1.0, -1.0, 1.0]
    n = MLP(3, [4, 4, 1])
    calulateBackwardPropagation(xs, ys, n)