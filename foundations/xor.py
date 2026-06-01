"""
Train a MLP to compute XOR
"""
from library.scalar.MLP import MLP as MLP_scalar
from library.tensor.MLP import MLP as MLP_tensor
import timeit


N_EPOCHS = 10_000
LEARNING_RATE = 0.05

inputs = [[1,1],[1,0],[0,1],[0,0]]
targets = [[0],[1],[1],[0]]


def update(model, loss):
    # Compute the gradient of each parameter in the model
    model.zero_grad()
    loss.backward()

    # Update the parameters with gradient descent
    for param in model.parameters():
        param.data -= LEARNING_RATE * param.grad


def scalar():
    model = MLP_scalar(2, [4,4,1])

    for it in range(N_EPOCHS):
        preds = [model(inp) for inp in inputs]
        loss = sum((targets[i][0] - preds[i][0])**2 for i in range(4))
        update(model, loss)
        # print(f"{it+1}. Loss: {loss}")
    # print(preds)


def tensor():
    model = MLP_tensor(2, [4,4,1])

    for it in range(N_EPOCHS):
        preds = model(inputs)
        loss = (targets - preds)**2
        update(model, loss)
        # print(f"{it+1}. Loss: {sum(loss.data)}")
    # print(preds)

print("scalar:", timeit.timeit(scalar, number=1))
print("tensor:", timeit.timeit(tensor, number=1))
