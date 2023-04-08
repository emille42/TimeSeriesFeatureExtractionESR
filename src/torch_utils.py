import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
import copy
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np


def show_progress_plots(all_metrics, metric_names, val=None):
    n_metrics = len(all_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, n_metrics))

    for i, metric_data, name in zip(range(n_metrics), all_metrics, metric_names):
        axes[i].plot(metric_data[0], label=f'{name}_train')
        if val is not None:
            axes[i].plot(metric_data[1], label=f'{name}_val')
        axes[i].legend()
    plt.show()


def train_model(model, loss_func, train_dataloader, optimizer, n_epochs, val_dataloader=None,
                device='cpu', metric=None, metric_params=None, logs=True, plots=True):
    model.to(device)

    best_model, best_model_train, best_model_val = None, None, None
    best_loss = best_loss_train = best_loss_val = 9999999
    best_metric = best_metric_train = best_metric_val = 9999999

    loss_train_history = []
    acc_train_history = []
    metric_train_history = []

    loss_val_history = []
    acc_val_history = []
    metric_val_history = []

    all_metrics = [[loss_train_history, loss_val_history],
                   [acc_train_history, acc_val_history],
                   [metric_train_history, metric_val_history]]
    metric_names = ['loss', 'accuracy', metric.__name__]

    try:
        for epoch in range(n_epochs):
            for phase in ['train', 'val']:
                if logs:
                    print(f"\nStart {phase} phase")
                if phase == 'train':
                    model.train()
                    dataloader = train_dataloader

                elif phase == 'val':
                    model.eval()
                    dataloader = val_dataloader

                    if dataloader is None:
                        continue

                epoch_loss = 0.
                epoch_acc = 0.
                epoch_metric = 0.

                for batch, labels in tqdm(dataloader, disable=not logs):
                    batch = batch.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        preds = model(batch)
                        loss_value = loss_func(preds, labels)
                        preds_class = preds.argmax(dim=1)

                        if phase == 'train':
                            loss_value.backward()
                            optimizer.step()

                # Metric values
                    epoch_loss += loss_value.item()
                    epoch_acc += (preds_class == labels.data).float().mean()
                    if metric:
                        epoch_metric += metric(labels.data.cpu(),
                                               preds_class.cpu(), average='macro')

                # Calc epoch metrics
                epoch_loss = np.around(epoch_loss/len(dataloader), decimals=3)
                epoch_acc = np.around(
                    float(epoch_acc/len(dataloader)), decimals=3)
                if metric:
                    epoch_metric = np.around(
                        float(epoch_metric/len(dataloader)), decimals=3)

                # Show epoch metrics in console
                if logs:
                    clear_output(True)
                    print(f"Epoch {epoch+1} {phase} loss: {epoch_loss}")
                    print(f"Epoch {epoch+1} {phase} acc: {epoch_acc}")
                if logs and metric:
                    print(f"Epoch {epoch+1} {phase} metric: {epoch_metric}")

                # Save epoch metrics and choose best models
                if phase == 'train':
                    loss_train_history.append(epoch_loss)
                    acc_train_history.append(epoch_acc)
                    if metric:
                        metric_train_history.append(epoch_metric)

                    if epoch_loss < best_loss_train:
                        best_model_train = copy.deepcopy(model)
                        best_loss_train = copy.deepcopy(epoch_loss)
                        if metric:
                            best_metric_train = copy.deepcopy(epoch_metric)
                        if logs:
                            print(
                                f"Find new best model on train with loss: {best_loss_train}! epoch: {epoch+1}")
                else:
                    loss_val_history.append(epoch_loss)
                    acc_val_history.append(epoch_acc)
                    if metric:
                        metric_val_history.append(epoch_metric)

                    if epoch_loss < best_loss_val:
                        best_model_val = copy.deepcopy(model)
                        best_loss_val = copy.deepcopy(epoch_loss)
                        if metric:
                            best_metric_val = copy.deepcopy(epoch_metric)
                        if logs:
                            print(
                                f"Find new best model on val with loss: {best_loss_val}, epoch: {epoch+1} !")

            # Choose best model
            if val_dataloader is not None:
                best_model = best_model_val
                best_loss = best_loss_val
                if metric:
                    best_metric = best_metric_val
            else:
                best_model = best_model_train
                best_loss = best_loss_train
                if metric:
                    best_metric = best_metric_train

    except KeyboardInterrupt:
        print("Training stopped by user!")
        print(f"Best train loss: {best_loss_train}")
        print(f"Best val loss: {best_loss_val}")
        print(f"Best train metric: {best_metric_train}")
        print(f"Best val metric: {best_metric_val}")
        if plots:
            show_progress_plots(all_metrics, metric_names, val_dataloader)
        return best_model, best_loss, best_metric

    if plots:
        show_progress_plots(all_metrics, metric_names, val_dataloader)
    print(f"Best train loss: {best_loss_train}")
    print(f"Best val loss: {best_loss_val}")
    print(f"Best train metric: {best_metric_train}")
    print(f"Best val metric: {best_metric_val}")
    return best_model, best_loss, best_metric


def get_efficientnet_v2_s_mod(n_classes=2, grad=False):
    model = models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    model.features[0][0] = nn.Conv2d(1, 24, kernel_size=(
        3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.features[0][0].dtype = torch.float32
    for param in model.parameters():
        param.requires_grad = grad
    model.classifier[1] = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.classifier[1].in_features, 64), nn.ReLU(),
                                        nn.BatchNorm1d(num_features=64), nn.Dropout(0.3), nn.Linear(64, n_classes))
    return model


def predict(model, dataloader, device='cpu'):
    model.eval()
    model.to(device)
    pred_labels = []

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            preds = model(inputs)
        pred_labels.append(nn.functional.softmax(
            preds, dim=1).argmax(-1).data.cpu().numpy())

    pred_labels = np.concatenate(pred_labels)
    clear_output(True)
    return pred_labels
