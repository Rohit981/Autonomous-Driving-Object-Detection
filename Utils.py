import matplotlib.pyplot as plt

  #Visualize Train and Test loss and Accuracy
def Visualize_loss_acc(train_loss,
                       val_loss, 
                       train_class_loss, 
                       val_class_loss, 
                       train_bbox_loss, 
                       val_bbox_loss, 
                       train_giou_loss, 
                       val_giou_loss):
    plt.figure(figsize=(12,5))
    epochs = range(1, len(train_loss) + 1)

    #Loss History
    plt.subplot(1,4,1)
    plt.plot(epochs,train_loss,label="Train Main Loss", color="blue")
    plt.plot(epochs,val_loss,label="Val Main loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Main Loss")
    plt.legend()
    plt.grid(True)

    #Loss Class History
    plt.subplot(1,4,2)
    plt.plot(epochs,train_class_loss,label="Train Class Loss", color="blue")
    plt.plot(epochs,val_class_loss,label="Val Class loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Class")
    plt.legend()
    plt.grid(True)

    #Loss BBOX History
    plt.subplot(1,4,3)
    plt.plot(epochs,train_bbox_loss,label="Train BBOX Loss", color="blue")
    plt.plot(epochs,val_bbox_loss,label="Val BBOX loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss BBOX")
    plt.legend()
    plt.grid(True)

    #Loss GIOU History
    plt.subplot(1,4,4)
    plt.plot(epochs,train_giou_loss,label="Train GIOU Loss", color="blue")
    plt.plot(epochs,val_giou_loss,label="Val GIOU loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss GIOU")
    plt.legend()
    plt.grid(True)

    plt.show()