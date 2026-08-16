# %% Import libraries
import logging
import os
import warnings
from operator import truediv

import hydra
import numpy as np
import spectral
import torch
import torchinfo
from hydra.core.hydra_config import HydraConfig
from scipy import io as sio
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

log = logging.getLogger(__name__)


# %% Load data
def loadData(name):
    data_path = os.path.join(os.getcwd(), "data")
    if name == "ip":
        data = sio.loadmat(os.path.join(data_path, "Indian_pines_corrected.mat"))[
            "indian_pines_corrected"
        ]
        labels = sio.loadmat(os.path.join(data_path, "Indian_pines_gt.mat"))[
            "indian_pines_gt"
        ]
    elif name == "sa":
        data = sio.loadmat(os.path.join(data_path, "Salinas_corrected.mat"))[
            "salinas_corrected"
        ]
        labels = sio.loadmat(os.path.join(data_path, "Salinas_gt.mat"))["salinas_gt"]
    elif name == "pu":
        data = sio.loadmat(os.path.join(data_path, "PaviaU.mat"))["paviaU"]
        labels = sio.loadmat(os.path.join(data_path, "PaviaU_gt.mat"))["paviaU_gt"]
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    return data, labels


# %% Split train and test set
def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testRatio, random_state=randomState, stratify=y
    )
    return X_train, X_test, y_train, y_test


def applyPCA(
    X, numComponents=75
):  # -> tuple[ndarray[tuple[int, int, int], dtype[Any]], PCA]:
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca


def padWithZeros(X, margin=2):
    newX = np.zeros(
        (X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]), dtype=np.float32
    )
    x_offset = margin
    y_offset = margin
    newX[x_offset : X.shape[0] + x_offset, y_offset : X.shape[1] + y_offset, :] = X
    return newX


def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros(
        (X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype=np.float32
    )
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]), dtype=np.int8)
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[
                r - margin : r + margin + 1, c - margin : c + margin + 1
            ]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


class HybridSN(nn.Module):
    def __init__(self, S, L, out):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3))

        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3))

        self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3))

        self.relu = nn.ReLU()

        H = S
        W = S
        D = L

        H = H - 3 + 1
        W = W - 3 + 1
        D = D - 7 + 1

        H = H - 3 + 1
        W = W - 3 + 1
        D = D - 5 + 1

        H = H - 3 + 1
        W = W - 3 + 1
        D = D - 3 + 1

        conv2d_in_channels = D * 32

        self.conv4 = nn.Conv2d(
            in_channels=conv2d_in_channels, out_channels=64, kernel_size=(3, 3)
        )

        H = H - 3 + 1
        W = W - 3 + 1

        flattened_size = H * W * 64

        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)

        self.dropout = nn.Dropout(0.4)

        self.output_layer = nn.Linear(128, out)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4))

        x = self.relu(self.conv4(x))

        x = torch.flatten(x, start_dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.output_layer(x)

        return x


TARGET_NAMES = {
    "ip": [
        "Alfalfa",
        "Corn-notill",
        "Corn-mintill",
        "Corn",
        "Grass-pasture",
        "Grass-trees",
        "Grass-pasture-mowed",
        "Hay-windrowed",
        "Oats",
        "Soybean-notill",
        "Soybean-mintill",
        "Soybean-clean",
        "Wheat",
        "Woods",
        "Buildings-Grass-Trees-Drives",
        "Stone-Steel-Towers",
    ],
    "sa": [
        "Brocoli_green_weeds_1",
        "Brocoli_green_weeds_2",
        "Fallow",
        "Fallow_rough_plow",
        "Fallow_smooth",
        "Stubble",
        "Celery",
        "Grapes_untrained",
        "Soil_vinyard_develop",
        "Corn_senesced_green_weeds",
        "Lettuce_romaine_4wk",
        "Lettuce_romaine_5wk",
        "Lettuce_romaine_6wk",
        "Lettuce_romaine_7wk",
        "Vinyard_untrained",
        "Vinyard_vertical_trellis",
    ],
    "pu": [
        "Asphalt",
        "Meadows",
        "Gravel",
        "Trees",
        "Painted metal sheets",
        "Bare Soil",
        "Bitumen",
        "Self-Blocking Bricks",
        "Shadows",
    ],
}


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def computeMetrics(y_true, y_pred, name):
    """OA / AA / Kappa / per-class accuracy. Cheap enough to run every epoch."""
    if name not in TARGET_NAMES:
        raise ValueError(f"Unknown dataset name: {name}")
    labels = np.arange(len(TARGET_NAMES[name]))

    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    each_acc, aa = AA_andEachClassAccuracy(confusion)

    return {
        "confusion": confusion,
        "oa": accuracy_score(y_true, y_pred) * 100,
        "each_acc": each_acc * 100,
        "aa": aa * 100,
        "kappa": cohen_kappa_score(y_true, y_pred) * 100,
    }


def reports(y_true, y_pred, name, test_loss=None):
    metrics = computeMetrics(y_true, y_pred, name)
    metrics["classification"] = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(TARGET_NAMES[name])),
        target_names=TARGET_NAMES[name],
        digits=4,
    )
    metrics["test_loss"] = test_loss
    return metrics


def predictModel(model, loader, criterion=None):
    """Run one inference pass, returning ground truth, predictions and mean loss."""
    model.eval()
    preds, targets = [], []
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc="Evaluating", leave=False):
            outputs = model(X_batch)
            if criterion is not None:
                loss_sum += criterion(outputs, y_batch).item() * X_batch.size(0)
            preds.append(outputs.argmax(1))
            targets.append(y_batch)
            n += X_batch.size(0)

    y_true = torch.cat(targets).to("cpu").numpy()
    y_pred = torch.cat(preds).to("cpu").numpy()
    test_loss = loss_sum / n if criterion is not None else None
    return y_true, y_pred, test_loss


def plot_outputs(y_true, y_pred, path):
    spectral.save_rgb(
        os.path.join(path, "ground_truth.png"),
        y_true,
        colors=spectral.spy_colors,
    )
    spectral.save_rgb(
        os.path.join(path, "predicted.png"),
        y_pred,
        colors=spectral.spy_colors,
    )


# %% Train the model
def trainModel(
    model,
    train_loader,
    test_loader,
    dataset_name,
    epochs=100,
    learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Mirrors tf.keras.optimizers.schedules.ExponentialDecay (staircase=False):
    # lr(t) = lr0 * decay_rate ** (t / decay_steps), where t counts optimizer steps.
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: decay_rate ** (step / decay_steps)
    )
    best = {"train_loss": 0.0, "test_loss": 0.0, "oa": 0.0, "aa": 0.0, "kappa": 0.0}
    for epoch in range(epochs):
        model.train()
        running_loss, n = 0.0, 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            running_loss += loss.item() * X_batch.size(0)
            n += X_batch.size(0)
        y_true, y_pred, test_loss = predictModel(model, test_loader, criterion)
        metrics = computeMetrics(y_true, y_pred, dataset_name)
        if metrics["oa"] > best["oa"]:
            best["train_loss"] = running_loss / n
            best["test_loss"] = test_loss
            best["oa"] = metrics["oa"]
            best["aa"] = metrics["aa"]
            best["kappa"] = metrics["kappa"]
            torch.save(
                model.state_dict(),
                os.path.join(
                    HydraConfig.get().runtime.output_dir,
                    f"model_best_{dataset_name}.pth",
                ),
            )
        tqdm.write(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"loss {running_loss / n:.4f} | test loss {test_loss:.4f} | "
            f"OA {metrics['oa']:.2f}% | AA {metrics['aa']:.2f}% | "
            f"Kappa {metrics['kappa']:.2f}% | LR {lr_scheduler.get_last_lr()[0]:.3e}"
        )
    torch.save(
        model.state_dict(),
        os.path.join(
            HydraConfig.get().runtime.output_dir, f"model_{dataset_name}_{epochs}.pth"
        ),
    )


# %% Main function
@hydra.main(config_path="./", config_name="config", version_base="1.3")
def main(config):
    torch.manual_seed(config.train.random_seed)
    dataset_name = config.train.dataset
    data, labels = loadData(dataset_name)
    log.info(
        f"Loaded {dataset_name} dataset with shape: {data.shape} and labels shape: {labels.shape}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    data_pca, _ = applyPCA(data, numComponents=config.train.pca_components)

    patches_data, patches_labels = createImageCubes(
        data_pca, labels, windowSize=config.train.window_size
    )

    X_train, X_test, y_train, y_test = splitTrainTestSet(
        patches_data, patches_labels, config.train.test_ratio, config.train.random_state
    )
    log.info(
        f"Split data into train and test sets with shapes: X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}"
    )

    X_train = X_train.reshape(
        -1,
        config.train.window_size,
        config.train.window_size,
        config.train.pca_components,
        1,
    )
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    log.info(f"Reshaped X_train to: {X_train.shape}")
    X_train = X_train.permute(0, 4, 3, 1, 2)
    log.info(f"Reshaped X_train to: {X_train.shape}")
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    log.info(f"Reshaped y_train to: {y_train.shape}")

    model = HybridSN(
        S=config.train.window_size,
        L=config.train.pca_components,
        out=int(torch.max(y_train) + 1),
    )

    torchinfo.summary(
        model,
        input_size=(
            1,
            1,
            config.train.pca_components,
            config.train.window_size,
            config.train.window_size,
        ),
        col_names=["input_size", "output_size", "num_params"],
        col_width=20,
        row_settings=["var_names"],
    )

    log.info(
        f"Initialized HybridSN model with parameters: S={config.train.window_size}, L={config.train.pca_components}, out={int(torch.max(y_train) + 1)}"
    )
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32)
        .to(device)
        .permute(0, 3, 1, 2)
        .unsqueeze(1),
        torch.tensor(y_test, dtype=torch.long).to(device),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    trainModel(
        model,
        train_loader,
        test_loader,
        dataset_name,
        epochs=config.train.epochs,
        learning_rate=config.train.learning_rate,
        decay_steps=config.train.lr_decay_steps,
        decay_rate=config.train.lr_decay_rate,
    )

    y_true, y_pred, test_loss = predictModel(
        model, test_loader, criterion=nn.CrossEntropyLoss()
    )
    metrics = reports(y_true, y_pred, dataset_name, test_loss=test_loss)

    full_patches, _ = createImageCubes(
        data_pca, labels, windowSize=config.train.window_size, removeZeroLabels=False
    )
    full_patches = full_patches.reshape(
        -1,
        config.train.window_size,
        config.train.window_size,
        config.train.pca_components,
        1,
    )
    full_patches = (
        torch.tensor(full_patches, dtype=torch.float32)
        .to(device)
        .permute(0, 4, 3, 1, 2)
    )
    full_loader = DataLoader(
        TensorDataset(
            full_patches, torch.zeros(full_patches.size(0), dtype=torch.long).to(device)
        ),
        batch_size=256,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    _, full_pred, _ = predictModel(model, full_loader)
    predicted_map = full_pred.reshape(labels.shape) + 1
    predicted_map[labels == 0] = 0
    plot_outputs(
        labels,
        predicted_map,
        path=os.path.join(HydraConfig.get().runtime.output_dir),
    )

    summary = "\n".join(
        [
            f"Dataset: {dataset_name}",
            f"Test loss: {metrics['test_loss']:.4f}",
            f"Overall accuracy (OA): {metrics['oa']:.2f}%",
            f"Average accuracy (AA): {metrics['aa']:.2f}%",
            f"Kappa: {metrics['kappa']:.2f}%",
            "",
            metrics["classification"],
            "Per-class accuracy:",
            *(
                f"  {cls_name}: {acc:.2f}%"
                for cls_name, acc in zip(
                    TARGET_NAMES[dataset_name], metrics["each_acc"]
                )
            ),
            "",
            "Confusion matrix:",
            np.array2string(metrics["confusion"], max_line_width=200),
        ]
    )

    log.info(summary)


# %%
main()
