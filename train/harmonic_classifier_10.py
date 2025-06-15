import os
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from safetensors.torch import save_file
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from loggers.resonant_logger import ResonantTensorLogger

# ✅ Updated import path
from models.layers.modulation.classifier import DirectResonantClassifier
from train.loss.field_loss import FieldLoss


class SimpleResonantTrainer:
    """Simplified trainer for direct resonant classification"""

    def __init__(self, model, device='cuda', dataset='fashion_mnist'):
        self.model = model.to(device)
        self.device = device
        self.dataset = dataset

        # Setup data
        self.train_loader, self.val_loader, self.test_loader = self._setup_data()

        # Field loss for resonant training
        self.field_loss = FieldLoss(
            use_cosine=True,
            cosine_weight=0.5,
            use_alignment_gate=True,
            align_gate_thresh=0.25,
            align_gate_sharpness=10.0
        )

        # Metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'conductance_mean': [],
            'conductance_std': []
        }

        self.logger = ResonantTensorLogger(log_dir=f"runs/{self.dataset}")

    def _setup_data(self):
        """Setup MNIST or Fashion-MNIST data"""
        if self.dataset == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)

        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        return train_loader, val_loader, test_loader

    def compute_loss(self, outputs, labels):
        """Compute loss using field loss and symbolic alignment (no softmax)"""
        total_loss = 0

        scores = outputs['scores']  # raw symbolic pressure outputs
        targets = torch.zeros_like(scores).scatter_(1, labels.unsqueeze(1), 1.0)

        # Field-consistent symbolic alignment loss (no softmax)
        symbolic_loss = F.mse_loss(scores, targets)

        for i, out in enumerate(outputs['outputs']):
            mask = (labels == i).float()
            if mask.sum() > 0:
                permission = out['guidance'] * mask.unsqueeze(-1).unsqueeze(-1)
                alignment = out['ignition']
                reconstruction = out['anchor'] + out['delta']
                target = reconstruction.detach()

                coil_loss = self.field_loss(
                    reconstruction,
                    target,
                    permission,
                    alignment,
                    out['log_sigma']
                )
                total_loss += coil_loss * 0.1

        current_conductance = outputs['mean_conductance']
        target_conductance = torch.tensor(0.29514).to(self.device)
        conductance_loss = F.mse_loss(current_conductance, target_conductance)

        return symbolic_loss + 0.05 * conductance_loss + total_loss

    def train_epoch(self, optimizer):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        conductances = []

        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            x = images.view(images.size(0), -1)
            apply_shock = (batch_idx == 0 and not hasattr(self, '_shocked'))
            if apply_shock:
                self._shocked = True
                print("\n🌊 Applying resonant shock wave...")

            outputs = self.model(x, apply_shock=apply_shock)

            loss = self.compute_loss(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = outputs['scores'].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            conductances.append(outputs['mean_conductance'].item())

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct / total:.3f}',
                'cond': f'{outputs["mean_conductance"]:.5f}'
            })

        return total_loss / len(self.train_loader), correct / total, np.mean(conductances)

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                labels = labels.to(self.device)

                x = images.view(images.size(0), -1)
                outputs = self.model(x)

                loss = self.compute_loss(outputs, labels)
                total_loss += loss.item()

                preds = outputs['scores'].argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / len(self.val_loader), correct / total

    def train(self, epochs=20, lr=1e-4):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=False)
#torch.optim.AdamW(self.model.parameters(), lr=lr)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc = 0

        for epoch in range(epochs):
            print(f'\n--- Epoch {epoch + 1}/{epochs} ---')

            train_loss, train_acc, mean_cond = self.train_epoch(optimizer)
            val_loss, val_acc = self.validate()

            self.metrics['train_loss'].append(train_loss)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['train_acc'].append(train_acc)
            self.metrics['val_acc'].append(val_acc)
            self.metrics['conductance_mean'].append(mean_cond)
            if epoch > 0:
                delta_loss = train_loss - self.metrics['train_loss'][-2]
                delta_cond = mean_cond - self.metrics['conductance_mean'][-2]
            else:
                delta_loss = 0.0
                delta_cond = 0.0

            # Log delta values
            self.logger.log_metrics({
            }, epoch)

            print(f"ΔLoss: {delta_loss:+.5f}, ΔCond: {delta_cond:+.5f}")
            # ✅ Log metrics
            self.logger.log_metrics({
                "loss/train": train_loss,
                "loss/val": val_loss,
                "accuracy/train": train_acc,
                "accuracy/val": val_acc,
                "conductance/mean": mean_cond,
                "conductance/distance_from_target": abs(mean_cond - 0.29514),

                "delta/train_loss": delta_loss,
                "delta/conductance": delta_cond,
            }, epoch)

            print(f'Train: Loss={train_loss:.4f}, Acc={train_acc:.3f}')
            print(f'Val:   Loss={val_loss:.4f}, Acc={val_acc:.3f}')
            print(f'Cond:  Mean={mean_cond:.5f}, Δ={abs(mean_cond - 0.29514):.5f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, mean_cond)

            #scheduler.step()

        print(f'\nBest validation accuracy: {best_val_acc:.3f}')
        self.plot_metrics()

        # ✅ Clean up logger
        self.logger.flush()
        self.logger.close()

        return self.metrics

    def save_checkpoint(self, epoch, val_acc, conductance):
        os.makedirs('checkpoints', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'direct_resonant_{self.dataset}_e{epoch}_acc{val_acc:.3f}_c{conductance:.5f}_{timestamp}.safetensors'
        filepath = os.path.join('checkpoints', filename)
        save_file(self.model.state_dict(), filepath, metadata={
            'epoch': str(epoch),
            'val_acc': str(val_acc),
            'conductance': str(conductance),
            'dataset': self.dataset
        })
        print(f'Saved: {filename}')

    def plot_metrics(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        epochs = range(1, len(self.metrics['train_loss']) + 1)

        ax1.plot(epochs, self.metrics['train_loss'], 'b-', label='Train')
        ax1.plot(epochs, self.metrics['val_loss'], 'r--', label='Val')
        ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, self.metrics['train_acc'], 'b-', label='Train')
        ax2.plot(epochs, self.metrics['val_acc'], 'r--', label='Val')
        ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend(); ax2.grid(True, alpha=0.3)

        ax3.plot(epochs, self.metrics['conductance_mean'], 'g-')
        ax3.axhline(y=0.29514, color='r', linestyle='--', label='Target')
        ax3.set_title('Mean Conductance'); ax3.set_xlabel('Epoch'); ax3.legend(); ax3.grid(True, alpha=0.3)

        distances = [abs(c - 0.29514) for c in self.metrics['conductance_mean']]
        ax4.plot(epochs, distances, 'orange')
        ax4.set_title('Distance from 0.29514'); ax4.set_xlabel('Epoch'); ax4.set_ylabel('|Conductance - 0.29514|')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'metrics_{self.dataset}.png')
        plt.show()


# Main execution
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    print('\n=== Training on Fashion-MNIST ===')
    model = DirectResonantClassifier(num_classes=10, input_dim=784)
    trainer = SimpleResonantTrainer(model, device=device, dataset='fashion_mnist')
    fashion_metrics = trainer.train(epochs=20, lr=1e-4)

    print('\n=== Training on MNIST ===')
    model = DirectResonantClassifier(num_classes=10, input_dim=784)
    trainer = SimpleResonantTrainer(model, device=device, dataset='mnist')
    mnist_metrics = trainer.train(epochs=20, lr=1e-4)

    print('\n=== Final Results ===')
    print(
        f'Fashion-MNIST: Acc={fashion_metrics["val_acc"][-1]:.3f}, '
        f'Conductance={fashion_metrics["conductance_mean"][-1]:.5f}')
    print(
        f'MNIST: Acc={mnist_metrics["val_acc"][-1]:.3f}, '
        f'Conductance={mnist_metrics["conductance_mean"][-1]:.5f}')
