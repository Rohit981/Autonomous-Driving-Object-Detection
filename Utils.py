import matplotlib.pyplot as plt

  #Visualize Train and Test loss and Accuracy
def Visualize_loss_acc(train_loss,val_loss):
    plt.figure(figsize=(12,5))
    epochs = range(1, len(train_loss) + 1)

    #Loss History
    plt.subplot(1,3,1)
    plt.plot(epochs,train_loss,label="Train Loss", color="blue")
    plt.plot(epochs,val_loss,label="Val_loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.legend()
    plt.grid(True)

    plt.show()