import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, 
                         train_f1_micro, val_f1_micro, train_f1_macro, val_f1_macro,
                         save_path=None):
    
    epochs = range(1, len(train_losses) + 1)

    #loss plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'r--', label='training loss')
    plt.plot(epochs, val_losses, 'r-', label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('training vs validation loss')
    plt.legend()
    plt.grid(True)

    #accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, [acc * 100 for acc in train_accuracies], 'b--', label='training accuracy')
    plt.plot(epochs, [acc * 100 for acc in val_accuracies], 'b-', label='validation accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('training vs validation accuracy')
    plt.legend()
    plt.grid(True)

    # F1 score plot
    plt.subplot(1, 3, 3)
    plt.plot(epochs, [f1 * 100 for f1 in train_f1_micro], 'g--', label='training F1 micro')
    plt.plot(epochs, [f1 * 100 for f1 in val_f1_micro], 'g-', label='validation F1 micro')
    plt.plot(epochs, [f1 * 100 for f1 in train_f1_macro], 'm--', label='training F1 macro')
    plt.plot(epochs, [f1 * 100 for f1 in val_f1_macro], 'm-', label='validation F1 macro')
    plt.xlabel('epoch')
    plt.ylabel('F1 score')
    plt.title('F1 micro and macro')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"saving plot to: {save_path}")
    else:
        plt.show()

