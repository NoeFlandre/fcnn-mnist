import pandas as pd
import matplotlib.pyplot as plt

try:
    train_df = pd.read_csv("train_metrics.csv")
    test_df = pd.read_csv("test_metrics.csv")

except FileNotFoundError :
    print("Files are missing")
    exit()

epochs = range(1, len(train_df)+1)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs, train_df['Train Loss'], color = 'red', label="Training Loss")
plt.plot(epochs, test_df['Test Loss'], color="blue", label = "Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.title("Losses with respect to the epochs")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, test_df['Accuracy'], color = "green", label ="Accuracy")
plt.title("Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

plt.savefig('mnist_metrics.png', dpi=300)
plt.close()