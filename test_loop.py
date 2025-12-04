import torch
import os
import csv

def test_loop(loss_fn, loader, model, device):

    if not os.path.exists("test_metrics.csv") :
        with open("test_metrics.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Test Loss', 'Accuracy'])

    model.eval()
    correct, test_loss = 0, 0
    num_batches = len(loader)
    size = len(loader.dataset)
    for batch, (X, Y) in enumerate(loader):
        X,Y=X.to(device), Y.to(device)
        Y_logits = model(X)
        loss = loss_fn(Y_logits, Y)
        test_loss += loss.item()
        Y_pred = Y_logits.argmax(1)

        correct += (Y_pred == Y).type(torch.float).sum().item()

    accuracy = correct / size # division by the normal of samples in the dataset
    test_loss /= num_batches # division by the number of batches in the dataset
    print(f"Test Loss : {test_loss} | Accuracy : {accuracy}")

    with open("test_metrics.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([test_loss, accuracy])