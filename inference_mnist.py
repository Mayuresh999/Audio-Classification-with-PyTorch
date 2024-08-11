import torch
from train import FeedForwardNetwork, download_mnist


class_mapping = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def predict(model, inputs, targets, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(inputs)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[targets]

    return predicted, expected

if __name__ == '__main__':
    feed_forward_net = FeedForwardNetwork()
    state_dict = torch.load("feed_forward_net.pth")
    feed_forward_net.load_state_dict(state_dict)
    _, val_data = download_mnist()
    input_data, target_data = val_data[0][0], val_data[0][1]
    predicted, expected = predict(feed_forward_net, input_data, target_data, class_mapping)

    print("Predicted", predicted, "expected", expected)
