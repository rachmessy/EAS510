import time
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

def train_multimodal2(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5, class_names=None):
    model.to(device)

    history = {
        'train_losses': [], 'val_losses': [],
        'train_accuracies': [], 'val_accuracies': [],
        'train_f1_micro': [], 'val_f1_micro': [],
        'train_f1_macro': [], 'val_f1_macro': []
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []

        for input_ids, attention_mask, image_input, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            image_input = image_input.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, image_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.numel()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        avg_train_loss = train_loss / len(train_loader)
        train_preds = torch.cat(all_preds).numpy()
        train_labels = torch.cat(all_labels).numpy()

        train_acc = correct / total
        train_f1_micro = f1_score(train_labels, train_preds, average='micro', zero_division=0)
        train_f1_macro = f1_score(train_labels, train_preds, average='macro', zero_division=0)

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for input_ids, attention_mask, image_input, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                image_input = image_input.to(device)
                labels = labels.to(device)

                outputs = model(input_ids, attention_mask, image_input)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                correct += (preds == labels).sum().item()
                total += labels.numel()

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        avg_val_loss = val_loss / len(val_loader)
        val_preds = torch.cat(all_preds).numpy()
        val_labels = torch.cat(all_labels).numpy()

        val_acc = correct / total
        val_f1_micro = f1_score(val_labels, val_preds, average='micro', zero_division=0)
        val_f1_macro = f1_score(val_labels, val_preds, average='macro', zero_division=0)

        print(
            f"epoch {epoch+1}/{num_epochs} | "
            f"train loss: {avg_train_loss:.4f}, acc: {train_acc:.4f}, F1 (macro): {train_f1_macro:.4f} | "
            f"val loss: {avg_val_loss:.4f}, acc: {val_acc:.4f}, F1 (macro): {val_f1_macro:.4f}"
        )

        if class_names is not None:
            print("Per-genre F1 scores:")
            for i, cls in enumerate(class_names):
                f1 = f1_score(val_labels[:, i], val_preds[:, i], zero_division=0)
                print(f"  {cls:15s}: {f1:.3f}")
            print()

        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(avg_val_loss)
        history['train_accuracies'].append(train_acc)
        history['val_accuracies'].append(val_acc)
        history['train_f1_micro'].append(train_f1_micro)
        history['val_f1_micro'].append(val_f1_micro)
        history['train_f1_macro'].append(train_f1_macro)
        history['val_f1_macro'].append(val_f1_macro)

    return history
