from library.MLP import MLP
from library.Value import Value


N_EPOCHS = 10_000

model = MLP(2, [4,4,1])

# We will train an MLP to learn XOR
inputs = [[1,1],[1,0],[0,1],[0,0]]
targets = [0,1,1,0]

for it in range(N_EPOCHS):
    # Compute the predictions of the model
    preds = [model(inp)[0] for inp in inputs]

    # Compute the loss
    loss = Value(0)
    for target, pred in zip(targets, preds):
        loss += (Value(target) - pred)**2

    # Compute the gradient of each parameter
    model.zero_grad()
    loss.backward()

    # Update the parameters with gradient descent
    for param in model.parameters():
        param.data = param.data - 0.05 * param.grad

    # print(f"{it+1}. Loss: {loss}")

print(preds)
