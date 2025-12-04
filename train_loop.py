import csv 
import os


def train_loop(optimizer, loss_fn, loader, model, device):

    if not os.path.exists("train_metrics.csv"):
        with open("train_metrics.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Train Loss'])

    model.train()
    train_loss = 0
    num_batches = len(loader)
    for batch, (X, Y) in enumerate(loader):
        X,Y=X.to(device), Y.to(device)
        Y_pred = model(X)
        loss = loss_fn(Y_pred, Y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= num_batches
    print(f"Training Loss : {train_loss}")
    with open("train_metrics.csv", "a", newline='') as f :
        writer = csv.writer(f)
        writer.writerow([train_loss])

            
