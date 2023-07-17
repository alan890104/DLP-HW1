import numpy as np
import matplotlib.pyplot as plt
from martinet.nn import Sequential, Linear
from martinet.act import Act
from martinet.loss import Loss
from martinet.optimizer import Optimizer
from cli import getParser


def generate_linear(n: int = 100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)

        if 0.1 * i == 0.5:
            continue

        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)


def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title("Ground truth", fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], "ro")
        else:
            plt.plot(x[i][0], x[i][1], "bo")

    plt.subplot(1, 2, 2)
    plt.title("Predict result", fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], "ro")
        else:
            plt.plot(x[i][0], x[i][1], "bo")

    plt.show()


def show_learning_curve(epoch, loss):
    plt.title("Learning Curve", fontsize=18)
    # plt.plot(epoch, loss, 'bo-', label='xxx')
    plt.plot(epoch, loss, "bo-", linewidth=1, markersize=2)
    plt.xlabel("epoch", fontsize=12)
    plt.ylabel("loss", fontsize=12)
    # plt.legend(loc = "best", fontsize=10)
    plt.show()


if __name__ == "__main__":
    # Parse arguments
    parser = getParser()
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Set dataset
    if args.dataset == "linear":
        X, y = generate_linear(args.sample_size)
    elif args.dataset == "xor":
        X, y = generate_XOR_easy()

    # Build model
    nn = Sequential(
        Linear(X.shape[1], args.hidden_size),
        Act("relu"),
        Linear(args.hidden_size, args.hidden_size),
        Act("relu"),
        Linear(args.hidden_size, args.output_size),
        Act("sigmoid"),
    )

    loss_fn = Loss(args.loss)
    opt = Optimizer(nn, kind=args.optimizer, lr=args.learning_rate)

    losses = []

    for epoch in range(args.epochs):
        pred_y = nn.forward(X)
        loss = loss_fn.forward(pred_y, y)
        pred_label = pred_y > 0.5
        accuracy = np.mean(pred_label == y)
        losses.append(loss)
        print(f"Epoch {epoch}, Loss: {loss:.7f}, Accuracy: {accuracy:.7f}")

        opt.zero_grad()
        grad = loss_fn.backward(pred_y, y)
        nn.backward(grad)
        opt.step()

    show_result(X, y, (pred_y > 0.5))
    show_learning_curve(range(len(losses)), losses)

    # testing model
    print("Testing model...")
    if args.dataset == "linear":
        X_test, y_test = generate_linear(args.sample_size)
    elif args.dataset == "xor":
        X_test, y_test = generate_XOR_easy()
    pred_y = nn.forward(X_test)
    pred_label = pred_y > 0.5
    accuracy = np.mean(pred_label == y_test)
    print(f"Test Accuracy: {accuracy:.7f}")
