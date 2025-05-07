import time
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs,
    scheduler=None,
    early_stop=None,
    class_names=None
):

    model = model.to(device)

    history = {
        'train_losses': [], 'val_losses': [],
        'train_accuracies': [], 'val_accuracies': [],
        'train_f1_micro': [], 'val_f1_micro': [],
        'train_f1_macro': [], 'val_f1_macro': []
    }

    best_val_macro = -float('inf')
    patience_left  = early_stop if early_stop is not None else None

    for epoch in range(num_epochs):
        start_time = time.time()

        # training
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        all_train_preds, all_train_labels = [], []

        for images, labels in tqdm(train_loader, desc=f"epoch {epoch+1}/{num_epochs} [training]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss   += loss.item() * labels.size(0)
            probs         = torch.sigmoid(logits)
            preds         = (probs > 0.4).float()
            train_correct += (preds == labels).sum().item()
            train_total   += labels.numel()

            all_train_preds.append(preds.cpu())
            all_train_labels.append(labels.cpu())

        train_loss /= len(train_loader.dataset)
        train_acc  = train_correct / train_total
        history['train_losses'].append(train_loss)
        history['train_accuracies'].append(train_acc)

        tr_preds  = torch.cat(all_train_preds).numpy()
        tr_labels = torch.cat(all_train_labels).numpy()
        history['train_f1_micro'].append(
            f1_score(tr_labels, tr_preds, average='micro', zero_division=0))
        history['train_f1_macro'].append(
            f1_score(tr_labels, tr_preds, average='macro', zero_division=0))

        # validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"epoch {epoch+1}/{num_epochs} [validation]"):
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss   = criterion(logits, labels)

                val_loss     += loss.item() * labels.size(0)
                probs         = torch.sigmoid(logits)
                preds         = (probs > 0.4).float()
                val_correct   += (preds == labels).sum().item()
                val_total     += labels.numel()

                all_val_preds.append(preds.cpu())
                all_val_labels.append(labels.cpu())

        val_loss /= len(val_loader.dataset)
        val_acc  = val_correct / val_total
        history['val_losses'].append(val_loss)
        history['val_accuracies'].append(val_acc)

        v_preds    = torch.cat(all_val_preds).numpy()
        v_labels   = torch.cat(all_val_labels).numpy()
        val_f1_micro = f1_score(v_labels, v_preds, average='micro', zero_division=0)
        val_f1_macro = f1_score(v_labels, v_preds, average='macro', zero_division=0)
        history['val_f1_micro'].append(val_f1_micro)
        history['val_f1_macro'].append(val_f1_macro)

        epoch_time = time.time() - start_time

        # logging
        print(
            f"epoch {epoch+1}/{num_epochs} | "
            f"time: {epoch_time:.1f}s | "
            f"training loss: {train_loss:.4f} | training accuracy: {train_acc:.4f} | "
            f"validation loss: {val_loss:.4f} | validation accuracy: {val_acc:.4f} | "
            f"validation macro-f1: {val_f1_macro:.4f}"
        )

        # per genre precision, recall and f1 score
        if class_names is not None:
            print("genre specific metrics:")
            for i, cls in enumerate(class_names):
                p = precision_score(v_labels[:, i], v_preds[:, i], zero_division=0)
                r = recall_score   (v_labels[:, i], v_preds[:, i], zero_division=0)
                f = f1_score       (v_labels[:, i], v_preds[:, i], zero_division=0)
                print(f"  {cls:15s}  P={p:.3f}, R={r:.3f},  F1={f:.3f}")
        print()

        # early stopping
        if early_stop is not None:
            if val_f1_macro > best_val_macro:
                best_val_macro = val_f1_macro
                patience_left  = early_stop
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"stopping early at epoch {epoch+1}. best validation macro-F1: {best_val_macro:.4f}")
                    break

        # scheduler
        if scheduler is not None:
            scheduler.step()

    return history
