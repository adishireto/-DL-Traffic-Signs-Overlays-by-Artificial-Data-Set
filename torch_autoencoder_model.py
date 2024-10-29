import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torchvision.utils as vutils
import time
from natsort import natsorted
import random
from functions import ensure_folder_exists, PairedDataset, save_images_from_loader, compute_ssim
import torch.multiprocessing as mp
from multiprocessing import freeze_support


# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 16, kernel_size=7, stride=1, padding='same'), nn.ReLU(),
                          nn.BatchNorm2d(16), nn.MaxPool2d(2, stride=2, padding=0)),   # 256 -> 128
            nn.Sequential(nn.Conv2d(16, 32, kernel_size=7, stride=1, padding='same'), nn.ReLU(),
                          nn.BatchNorm2d(32), nn.MaxPool2d(2, stride=2, padding=0)),   # 128 -> 64
            nn.Sequential(nn.Conv2d(32, 64, kernel_size=7, stride=1, padding='same'), nn.ReLU(),
                          nn.BatchNorm2d(64), nn.MaxPool2d(2, stride=2, padding=0)),   # 64 -> 32
            nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same'), nn.ReLU(),
                          nn.BatchNorm2d(128), nn.MaxPool2d(2, stride=2, padding=0)),  # 32 -> 16
            nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same'), nn.ReLU(),
                          nn.BatchNorm2d(256), nn.MaxPool2d(2, stride=2, padding=0)),  # 16 -> 8
            nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding='same'), nn.ReLU(),
                          nn.BatchNorm2d(512), nn.MaxPool2d(2, stride=2, padding=0)),   # 8 -> 4
            nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding='same'), nn.ReLU(),
                          nn.BatchNorm2d(1024), nn.MaxPool2d(2, stride=2, padding=0))   # 4 -> 2
        ])
        # Decoder
        self.decoder = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding='same'), nn.ReLU(),
                          nn.BatchNorm2d(512)),  # 2 -> 4
            nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding='same'), nn.ReLU(),
                          nn.BatchNorm2d(256)),  # 4 -> 8
            nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding='same'), nn.ReLU(),
                          nn.BatchNorm2d(128)),  # 8 -> 16
            nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding='same'), nn.ReLU(),
                          nn.BatchNorm2d(64)),  # 16 -> 32
            nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding='same'), nn.ReLU(),
                          nn.BatchNorm2d(32)),  # 32 -> 64
            nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, stride=1, padding='same'), nn.ReLU(),
                          nn.BatchNorm2d(16)),  # 64 -> 128
            nn.Sequential(nn.Conv2d(16, 3, kernel_size=2, stride=2), nn.Sigmoid())  # 128 -> 256
        ])

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)

        skips = skips[::-1][1:]  # Reverse and remove the last element

        for i, layer in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode='nearest')  # Use F.interpolate from torch.nn.functional
            x = layer(x)
            if i < len(skips):
                x += skips[i]

        x = F.interpolate(x, scale_factor=2, mode='nearest')  # Use F.interpolate from torch.nn.functional
        return x


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, prefix='image'):
        super().__init__(root, transform)
        self.prefix = prefix
        self.samples = natsorted(
            self.samples,
            key=lambda x: (
                x[1],  # Sort by class label first (keeps class ordering)
                int(os.path.basename(x[0]).replace(self.prefix, '').split('.')[0])  # Sort by numeric part of the filename
            )
        )
        self.imgs = self.samples
        self.imgs = self.samples


def visualize_training(masks, images, outputs, epoch, save_path):
    """
    Save and display a batch of masks, target images, and outputs.

    Args:
        masks (torch.Tensor): The input masks.
        images (torch.Tensor): The input images.
        outputs (torch.Tensor): The output images.
        epoch (int): The current epoch.
        save_path (str): The path to save the visualization.

    Returns:
        None
    """
    # Unormalize and save images for visualization
    masks_grid = vutils.make_grid(masks.cpu(), normalize=True, scale_each=True)
    images_grid = vutils.make_grid(images.cpu(), normalize=True, scale_each=True)
    outputs_grid = vutils.make_grid(outputs.cpu(), normalize=True, scale_each=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(masks_grid.permute(1, 2, 0))
    axes[0].set_title('Masks')
    axes[0].axis('off')

    axes[1].imshow(images_grid.permute(1, 2, 0))
    axes[1].set_title('Target Images')
    axes[1].axis('off')

    axes[2].imshow(outputs_grid.permute(1, 2, 0))
    axes[2].set_title('Reconstructed Images')
    axes[2].axis('off')

    plt.suptitle(f'Epoch {epoch + 1}')
    plt.show()

    fig.savefig(os.path.join(save_path, f'epoch_{epoch + 1}.png'))
    plt.close(fig)


def save_parameters(lr, step_size, gamma, batch_size, kernel_size, file_path, alpha, beta):
    """
    Save the given parameters to a file.

    Args:
        lr (float): The learning rate.
        step_size (int): The step size.
        gamma (float): The gamma value.
        batch_size (int): The batch size.
        kernel_size (int): The kernel size.
        file_path (str): The path to the file.
        alpha (float): The alpha value.
        beta (float): The beta value.

    Returns:
        None
    """
    with open(file_path, 'w') as f:
        f.write(f'Learning Rate (lr): {lr}\n')
        f.write(f'Step Size (step_size): {step_size}\n')
        f.write(f'Gamma (gamma): {gamma}\n')
        f.write(f'Batch Size (batch_size): {batch_size}\n')
        f.write(f'Kernel Size (kernel_size): {kernel_size}\n')
        f.write(f'alpha Size (alpha_size): {alpha}\n')
        f.write(f'beta Size (beta_size): {beta}\n')

def save_epoch_result(epoch, num_epochs, epoch_loss, epoch_ssim, val_loss, val_ssim, file_path):
    """
    Save the results of an epoch to a file.

    Args:
        epoch (int): The current epoch.
        num_epochs (int): The total number of epochs.
        epoch_loss (float): The loss value for the current epoch.
        epoch_ssim (float): The SSIM value for the current epoch.
        val_loss (float): The loss value for the validation set.
        val_ssim (float): The SSIM value for the validation set.
        file_path (str): The path to the file where the results will be saved.

    Returns:
        None
    """
    with open(file_path, 'a') as f:
        f.write(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.8f}, Training SSIM: {epoch_ssim:.4f}\n')
        f.write(f'Validation Loss: {val_loss:.8f}, Validation SSIM: {val_ssim:.4f}\n')


def split_dataset(dataset, train_ratio=0.7, valid_ratio=0.2):
    """
    Split a dataset into training, validation, and testing subsets.

    Parameters:
        dataset (Dataset): The dataset to be split.
        train_ratio (float, optional): The ratio of samples to be used for training. Defaults to 0.7.
        valid_ratio (float, optional): The ratio of samples to be used for validation. Defaults to 0.2.

    Returns:
        Tuple[Subset, Subset, Subset]: A tuple containing the training dataset, validation dataset, and testing dataset.
    """
    total_samples = len(dataset)
    train_size = int(train_ratio * total_samples)
    valid_size = int(valid_ratio * total_samples)
    test_size = total_samples - train_size - valid_size

    test_indices = list(range(test_size))
    valid_indices = list(range(test_size, test_size + valid_size))
    train_indices = list(range(test_size + valid_size, total_samples))

    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, valid_dataset, test_dataset


def main():
    """
    Trains an autoencoder model using the given dataset.

    This function trains an autoencoder model using the provided dataset. It sets up the necessary paths, 
    defines the autoencoder model, loads the datasets, and creates the necessary DataLoaders. 
    The function then trains the model, evaluates the training and validation performance, and saves the results. 
    Finally, it saves the loss and SSIM graphs.

    Parameters:
        None

    Returns:
        None
    """
    mp.set_start_method('spawn', force=True)
    start_time = time.time()

    # Path to your dataset folders
    files_path = 'C:/Users/alontsa/Desktop/dataset'
    ensure_folder_exists(files_path)
    masks_dir = f'{files_path}/cropped_masks'
    ensure_folder_exists(masks_dir)
    images_dir = f'{files_path}/cropped_images'
    ensure_folder_exists(images_dir)
    autoencoder_path = 'C:/Users/alontsa/Desktop/dataset/models/autoencoder'
    model_name = 'torch_autoencoder_interpolation_LANCZOS'
    ensure_folder_exists(f'{autoencoder_path}/{model_name}')
    train_results_save_path = os.path.join(f'{autoencoder_path}/{model_name}', 'train_results.txt')
    sign = 'final1'
    # Initialize the model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    print(device)
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()
    alpha, beta = 0.6, 0.4
    lr, step_size, gamma, batch_size = 1e-3, 10, 0.1, 32
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    img_height, img_width = 256, 256

    # Save parameters
    param_save_path = os.path.join(f'{autoencoder_path}/{model_name}', 'training_parameters.txt')
    save_parameters(lr=lr, step_size=step_size, gamma=gamma, batch_size=batch_size, kernel_size=7,
                    file_path=param_save_path, alpha=alpha, beta=beta)

    # Use torchvision's transforms to preprocess the images
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor()
    ])

    # Load the datasets
    color_block_dataset = CustomImageFolder(root=masks_dir, transform=transform, prefix='mask')
    traffic_sign_dataset = CustomImageFolder(root=images_dir, transform=transform, prefix='image')

    # Create paired datasets using the custom paired_generator
    paired_dataset = PairedDataset(color_block_dataset, traffic_sign_dataset)

    # Group indices by class
    class_indices = {class_name: [] for class_name in paired_dataset.get_class_names()}
    for i in range(len(paired_dataset)):
        _, _, label = paired_dataset[i]
        class_name = paired_dataset.get_class_names()[label]
        class_indices[class_name].append(i)

    # Split indices for each class
    split_indices = {'train': [], 'valid': [], 'test': []}
    for class_name, indices in class_indices.items():
        n = len(indices)
        train_size = int(0.7 * n)
        valid_size = int(0.2 * n)
        test_size = n - train_size - valid_size
        split_indices['test'].extend(indices[:test_size])
        split_indices['valid'].extend(indices[test_size:test_size + valid_size])
        split_indices['train'].extend(indices[test_size + valid_size:])

    random.shuffle(split_indices['valid'])
    # Create datasets for each split
    train_dataset = torch.utils.data.Subset(paired_dataset, split_indices['train'])
    valid_dataset = torch.utils.data.Subset(paired_dataset, split_indices['valid'])
    test_dataset = torch.utils.data.Subset(paired_dataset, split_indices['test'])

    # Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    save_path = os.path.join(files_path, f'model_test_{sign}')
    save_images_from_loader(test_loader, save_path)

    # Visualize the first pair of images
    first_pair = paired_dataset[0]
    img1, img2, _ = first_pair  # Discard the label
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img1.permute(1, 2, 0))
    axes[0].set_title("Mask")
    axes[0].axis('off')
    axes[1].imshow(img2.permute(1, 2, 0))
    axes[1].set_title("Image")
    axes[1].axis('off')
    plt.show()

    # Train the model
    num_epochs = 100
    early_stopping_patience = 50
    best_loss = float('inf')
    early_stopping_counter = 0
    epoch_loss_list = []
    train_ssim_list = []
    val_loss_list = []
    val_ssim_list = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for masks, images, _ in train_loader:
            masks = masks.to(device)
            images = images.to(device)

            outputs = model(masks)
            # Compute combined loss
            loss_l1 = criterion_l1(outputs, images)
            loss_mse = criterion_mse(outputs, images)
            loss = alpha * loss_l1 + beta * loss_mse  # You can adjust the weight of each loss here

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        # Evaluation of training performance
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_ssim = compute_ssim(images, outputs)  # Compute SSIM for the last batch of training data
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.8f}, Training SSIM: {epoch_ssim:.4f}')
        epoch_loss_list.append(epoch_loss)
        train_ssim_list.append(epoch_ssim)

        # Validation step
        model.eval()
        val_loss = 0.0
        val_ssim = 0.0

        with torch.no_grad():
            for masks, images, _ in valid_loader:
                masks = masks.to(device)
                images = images.to(device)
                outputs = model(masks)
                # Compute combined loss
                loss_l1 = criterion_l1(outputs, images)
                loss_mse = criterion_mse(outputs, images)
                loss = alpha * loss_l1 + beta * loss_mse  # You can adjust the weight of each loss here
                val_loss += loss.item() * images.size(0)
                val_ssim += compute_ssim(images, outputs) * images.size(0)

        # Evaluation of Validation  performance
        val_loss /= len(valid_loader.dataset)
        val_ssim /= len(valid_loader.dataset)
        print(f'Validation Loss: {val_loss:.8f}, Validation SSIM: {val_ssim:.4f}')
        val_loss_list.append(val_loss)
        val_ssim_list.append(val_ssim)

        save_epoch_result(epoch=epoch, num_epochs=num_epochs, epoch_loss=epoch_loss, epoch_ssim=epoch_ssim,
                          val_loss=val_loss, val_ssim=val_ssim, file_path=train_results_save_path)

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                for masks, images, _ in valid_loader:
                    masks = masks.to(device)
                    images = images.to(device)
                    outputs = model(masks)
                    visualize_training(masks, images, outputs, epoch, f'{autoencoder_path}/{model_name}')
                    break

        if val_loss < best_loss:
            best_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), os.path.join(autoencoder_path, f'model_{model_name}_2024.pth'))
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping")
            break

        scheduler.step()

    # Plot the loss and SSIM graphs
    plt.figure()
    plt.plot(range(1, epoch + 2), epoch_loss_list, label='Training Loss')
    plt.plot(range(1, epoch + 2), val_loss_list, label='Validation Loss')
    plt.title('Autoencoder Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(f'{autoencoder_path}/{model_name}', 'Autoencoder_Loss.png'))
    plt.show()

    plt.figure()
    plt.plot(range(1, epoch + 2), train_ssim_list, label='Training SSIM')
    plt.plot(range(1, epoch + 2), val_ssim_list, label='Validation SSIM')
    plt.title('Autoencoder SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.savefig(os.path.join(f'{autoencoder_path}/{model_name}', 'Autoencoder_SSIM.png'))
    plt.show()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The program took {elapsed_time} seconds to run.")


if __name__ == '__main__':
    freeze_support()
    main()
