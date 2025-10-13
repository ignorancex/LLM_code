"""
Train improved PPG+ECG model - Using Cross-Attention
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report, f1_score
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import gc
import argparse
import yaml

from multimodal_model_crossattn import ImprovedMultiModalSleepNet
from multimodal_dataset_aligned import get_dataloaders


class CrossAttentionTrainer:
    def __init__(self, config, run_id=None):
        self.config = config
        self.run_id = run_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create output directories
        self.setup_directories()

        # Initialize tensorboard
        self.writer = SummaryWriter(self.log_dir)

        # Save configuration
        with open(os.path.join(self.checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        # Mixed precision training
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
            print("Using mixed precision training")

    def setup_directories(self):
        """Create necessary directories"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"crossattn_ecg_v2_{timestamp}"
        if self.run_id is not None:
            model_name += f"_run{self.run_id}"

        self.output_dir = os.path.join(self.config['output']['save_dir'], model_name)
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        self.results_dir = os.path.join(self.output_dir, 'results')

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def calculate_class_weights(self, train_dataset):
        """Calculate class weights"""
        print("\nCalculating class weights...")

        all_labels = []
        sample_size = min(len(train_dataset), 50)

        for idx in tqdm(range(sample_size), desc="Sampling labels"):
            _, _, labels = train_dataset[idx]
            valid_labels = labels[labels != -1]
            all_labels.extend(valid_labels.numpy().tolist())

        label_counts = Counter(all_labels)
        class_counts = [label_counts.get(i, 1) for i in range(4)]
        total_samples = sum(class_counts)

        print(f"\nLabel distribution:")
        stage_names = ['Wake', 'Light', 'Deep', 'REM']
        for i, count in enumerate(class_counts):
            percentage = count / total_samples * 100
            print(f"  {stage_names[i]}: {count} samples ({percentage:.2f}%)")

        # Use inverse frequency weighting
        class_weights = torch.tensor([total_samples / (4 * count) for count in class_counts],
                                     dtype=torch.float32)

        print(f"\nClass weights: {class_weights}")

        return class_weights.to(self.device)

    def train_epoch(self, model, dataloader, optimizer, criterion, scheduler, epoch):
        """Train one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, (ppg, ecg, labels) in enumerate(pbar):
            ppg = ppg.to(self.device)
            ecg = ecg.to(self.device)
            labels = labels.to(self.device)

            # Mixed precision training
            if self.use_amp:
                with autocast():
                    outputs = model(ppg, ecg)
                    outputs_reshaped = outputs.permute(0, 2, 1)
                    loss = criterion(
                        outputs_reshaped.reshape(-1, 4),
                        labels.reshape(-1)
                    )

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

            else:
                # Standard training
                outputs = model(ppg, ecg)
                outputs_reshaped = outputs.permute(0, 2, 1)
                loss = criterion(
                    outputs_reshaped.reshape(-1, 4),
                    labels.reshape(-1)
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # Update learning rate
            if scheduler is not None:
                scheduler.step()

            # Statistics
            mask = labels != -1
            valid_outputs = outputs_reshaped[mask]
            valid_labels = labels[mask]

            if valid_labels.numel() > 0:
                _, predicted = valid_outputs.max(1)
                correct += predicted.eq(valid_labels).sum().item()
                total += valid_labels.numel()
                running_loss += loss.item() * valid_labels.numel()

            # Update progress bar
            if total > 0:
                pbar.set_postfix({
                    'loss': running_loss / total,
                    'acc': correct / total,
                    'lr': optimizer.param_groups[0]['lr']
                })

            # Periodic memory cleanup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        epoch_loss = running_loss / total if total > 0 else 0
        epoch_acc = correct / total if total > 0 else 0

        return epoch_loss, epoch_acc

    def validate(self, model, dataloader, criterion):
        """Validate with per-patient median metrics"""
        model.eval()
        running_loss = 0.0

        # For overall calculation
        all_preds = []
        all_labels = []

        # For per-patient calculation
        patient_predictions = defaultdict(list)
        patient_labels = defaultdict(list)

        with torch.no_grad():
            for batch_idx, (ppg, ecg, labels) in enumerate(tqdm(dataloader, desc="Validation")):
                ppg = ppg.to(self.device)
                ecg = ecg.to(self.device)
                labels = labels.to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = model(ppg, ecg)
                        outputs_reshaped = outputs.permute(0, 2, 1)
                        loss = criterion(
                            outputs_reshaped.reshape(-1, 4),
                            labels.reshape(-1)
                        )
                else:
                    outputs = model(ppg, ecg)
                    outputs_reshaped = outputs.permute(0, 2, 1)
                    loss = criterion(
                        outputs_reshaped.reshape(-1, 4),
                        labels.reshape(-1)
                    )

                # Process each sample in batch
                batch_size = outputs.shape[0]
                for i in range(batch_size):
                    patient_idx = batch_idx * dataloader.batch_size + i

                    # Get valid predictions and labels for current patient
                    mask = labels[i] != -1
                    if mask.any():
                        patient_outputs = outputs_reshaped[i][mask]
                        patient_labels_i = labels[i][mask]

                        _, predicted = patient_outputs.max(1)

                        # Store per-patient data
                        patient_predictions[patient_idx].extend(predicted.cpu().numpy())
                        patient_labels[patient_idx].extend(patient_labels_i.cpu().numpy())

                        # Also save to overall lists
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(patient_labels_i.cpu().numpy())

                        running_loss += loss.item() * patient_labels_i.numel()

        # Calculate per-patient metrics
        patient_accuracies = []
        patient_kappas = []
        patient_f1s = []

        for patient_idx in patient_predictions:
            if len(patient_predictions[patient_idx]) > 0:
                patient_acc = np.mean(np.array(patient_predictions[patient_idx]) ==
                                      np.array(patient_labels[patient_idx]))
                patient_accuracies.append(patient_acc)

                # Only calculate kappa when patient has multiple classes
                unique_labels = np.unique(patient_labels[patient_idx])
                if len(unique_labels) > 1:
                    patient_kappa = cohen_kappa_score(patient_labels[patient_idx],
                                                      patient_predictions[patient_idx])
                    patient_kappas.append(patient_kappa)

                patient_f1 = f1_score(patient_labels[patient_idx],
                                      patient_predictions[patient_idx],
                                      average='weighted')
                patient_f1s.append(patient_f1)

        # Calculate metrics
        # Overall metrics
        epoch_loss = running_loss / len(all_labels) if all_labels else 0
        overall_accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) if all_labels else 0
        overall_kappa = cohen_kappa_score(all_labels, all_preds) if all_labels else 0
        overall_f1 = f1_score(all_labels, all_preds, average='weighted') if all_labels else 0

        # Per-patient median metrics
        median_accuracy = np.median(patient_accuracies) if patient_accuracies else 0
        median_kappa = np.median(patient_kappas) if patient_kappas else 0
        median_f1 = np.median(patient_f1s) if patient_f1s else 0

        # Print per-patient kappa distribution
        if patient_kappas:
            print(f"\nPer-patient Kappa distribution:")
            print(f"  Min: {np.min(patient_kappas):.4f}")
            print(f"  25%: {np.percentile(patient_kappas, 25):.4f}")
            print(f"  Median: {median_kappa:.4f}")
            print(f"  75%: {np.percentile(patient_kappas, 75):.4f}")
            print(f"  Max: {np.max(patient_kappas):.4f}")

        # Calculate confusion matrix and per-class metrics
        cm = confusion_matrix(all_labels, all_preds)
        per_class_metrics = self.calculate_per_class_metrics(cm)

        return {
            'loss': epoch_loss,
            'overall_accuracy': overall_accuracy,
            'overall_kappa': overall_kappa,
            'overall_f1': overall_f1,
            'median_accuracy': median_accuracy,
            'median_kappa': median_kappa,
            'median_f1': median_f1,
            'all_preds': all_preds,
            'all_labels': all_labels,
            'per_class_metrics': per_class_metrics,
            'patient_kappas': patient_kappas,
            'confusion_matrix': cm
        }

    def calculate_per_class_metrics(self, cm):
        """Calculate per-class metrics"""
        n_classes = cm.shape[0]
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        f1 = np.zeros(n_classes)

        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp

            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) \
                if (precision[i] + recall[i]) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def train(self):
        """Main training loop"""
        print(f"\n{'=' * 60}")
        print("Training Improved PPG+ECG Model with Cross-Attention V2")
        print(f"{'=' * 60}")

        # Prepare data
        data_paths = {
            'ppg': self.config['data']['ppg_file'],
            'real_ecg': self.config['data']['ecg_file'],
            'index': self.config['data']['index_file']
        }

        # Create data loaders
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_dataloaders(
            data_paths,
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            model_type='multimodal',
            use_sleepppg_test_set=True
        )

        # Create model
        model = ImprovedMultiModalSleepNet(
            n_classes=4,
            d_model=self.config['model']['d_model'],
            n_heads=self.config['model']['n_heads'],
            n_fusion_blocks=self.config['model']['n_fusion_blocks']
        ).to(self.device)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Calculate class weights
        class_weights = self.calculate_class_weights(train_dataset)
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        # Learning rate scheduler
        total_steps = len(train_loader) * self.config['training']['num_epochs']
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config['training']['learning_rate'],
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )

        # Training loop
        best_kappa = 0
        best_epoch = 0
        patience_counter = 0

        train_losses = []
        val_losses = []
        val_overall_kappas = []
        val_median_kappas = []

        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch}/{self.config['training']['num_epochs']}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, scheduler, epoch
            )
            train_losses.append(train_loss)

            # Validate
            val_results = self.validate(model, val_loader, criterion)
            val_losses.append(val_results['loss'])
            val_overall_kappas.append(val_results['overall_kappa'])
            val_median_kappas.append(val_results['median_kappa'])

            print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_results['loss']:.4f}")
            print(f"Val Overall - Acc: {val_results['overall_accuracy']:.4f}, "
                  f"Kappa: {val_results['overall_kappa']:.4f}, F1: {val_results['overall_f1']:.4f}")
            print(f"Val Median  - Acc: {val_results['median_accuracy']:.4f}, "
                  f"Kappa: {val_results['median_kappa']:.4f}, F1: {val_results['median_f1']:.4f}")

            # Print per-class performance
            stage_names = ['Wake', 'Light', 'Deep', 'REM']
            print("\nPer-class performance:")
            for i, name in enumerate(stage_names):
                print(f"  {name}: P={val_results['per_class_metrics']['precision'][i]:.3f}, "
                      f"R={val_results['per_class_metrics']['recall'][i]:.3f}, "
                      f"F1={val_results['per_class_metrics']['f1'][i]:.3f}")

            # Log to tensorboard
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Train/Acc', train_acc, epoch)
            self.writer.add_scalar('Val/Loss', val_results['loss'], epoch)
            self.writer.add_scalar('Val/Overall_Accuracy', val_results['overall_accuracy'], epoch)
            self.writer.add_scalar('Val/Overall_Kappa', val_results['overall_kappa'], epoch)
            self.writer.add_scalar('Val/Overall_F1', val_results['overall_f1'], epoch)
            self.writer.add_scalar('Val/Median_Accuracy', val_results['median_accuracy'], epoch)
            self.writer.add_scalar('Val/Median_Kappa', val_results['median_kappa'], epoch)
            self.writer.add_scalar('Val/Median_F1', val_results['median_f1'], epoch)
            self.writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

            # Log per-class metrics
            for i, name in enumerate(stage_names):
                self.writer.add_scalar(f'Val/Precision_{name}',
                                       val_results['per_class_metrics']['precision'][i], epoch)
                self.writer.add_scalar(f'Val/Recall_{name}',
                                       val_results['per_class_metrics']['recall'][i], epoch)
                self.writer.add_scalar(f'Val/F1_{name}',
                                       val_results['per_class_metrics']['f1'][i], epoch)

            # Save best model (based on overall kappa)
            if val_results['overall_kappa'] > best_kappa:
                best_kappa = val_results['overall_kappa']
                best_epoch = epoch
                patience_counter = 0

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_overall_kappa': best_kappa,
                    'best_median_kappa': val_results['median_kappa'],
                    'val_acc': val_results['overall_accuracy'],
                    'val_f1': val_results['overall_f1'],
                    'config': self.config
                }

                best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                torch.save(checkpoint, best_path)
                print(f"Saved best model with overall kappa: {best_kappa:.4f}")

                # Save confusion matrix
                self.plot_confusion_matrix(val_results['confusion_matrix'], epoch)
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.config['training']['patience']:
                print(f"\nEarly stopping at epoch {epoch}")
                break

            # Periodic save
            if epoch % self.config['output']['save_frequency'] == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, checkpoint_path)

            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()

        print(f"\nBest validation overall kappa: {best_kappa:.4f} at epoch {best_epoch}")

        # Test on test set
        print("\n" + "=" * 60)
        print("Evaluating on test set...")
        print("=" * 60)

        # Load best model
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])

        # Test
        test_results = self.validate(model, test_loader, criterion)

        print(f"\nTest Results:")
        print(f"  Overall - Accuracy: {test_results['overall_accuracy']:.4f}, "
              f"Kappa: {test_results['overall_kappa']:.4f}, F1: {test_results['overall_f1']:.4f}")
        print(f"  Median  - Accuracy: {test_results['median_accuracy']:.4f}, "
              f"Kappa: {test_results['median_kappa']:.4f}, F1: {test_results['median_f1']:.4f}")

        # Detailed report
        report = classification_report(
            test_results['all_labels'], test_results['all_preds'],
            target_names=['Wake', 'Light', 'Deep', 'REM'],
            output_dict=True
        )

        print("\nClassification Report:")
        print(classification_report(
            test_results['all_labels'], test_results['all_preds'],
            target_names=['Wake', 'Light', 'Deep', 'REM']
        ))

        # Save results
        results = {
            'test_accuracy_overall': test_results['overall_accuracy'],
            'test_kappa_overall': test_results['overall_kappa'],
            'test_f1_overall': test_results['overall_f1'],
            'test_accuracy_median': test_results['median_accuracy'],
            'test_kappa_median': test_results['median_kappa'],
            'test_f1_median': test_results['median_f1'],
            'test_loss': test_results['loss'],
            'best_epoch': best_epoch,
            'classification_report': report,
            'confusion_matrix': test_results['confusion_matrix'].tolist(),
            'per_class_metrics': {
                'precision': test_results['per_class_metrics']['precision'].tolist(),
                'recall': test_results['per_class_metrics']['recall'].tolist(),
                'f1': test_results['per_class_metrics']['f1'].tolist()
            },
            'patient_kappa_stats': {
                'min': float(np.min(test_results['patient_kappas'])) if test_results['patient_kappas'] else 0,
                'max': float(np.max(test_results['patient_kappas'])) if test_results['patient_kappas'] else 0,
                'mean': float(np.mean(test_results['patient_kappas'])) if test_results['patient_kappas'] else 0,
                'std': float(np.std(test_results['patient_kappas'])) if test_results['patient_kappas'] else 0,
                'median': float(test_results['median_kappa']),
                '25_percentile': float(np.percentile(test_results['patient_kappas'], 25)) if test_results[
                    'patient_kappas'] else 0,
                '75_percentile': float(np.percentile(test_results['patient_kappas'], 75)) if test_results[
                    'patient_kappas'] else 0
            },
            'config': self.config
        }

        with open(os.path.join(self.results_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        # Plot final confusion matrix
        self.plot_confusion_matrix(test_results['confusion_matrix'], 'final')

        # Plot training curves
        self.plot_training_curves(train_losses, val_losses, val_overall_kappas, val_median_kappas)

        self.writer.close()

        return results

    def plot_confusion_matrix(self, cm, epoch):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))

        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Create annotations
        annotations = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

        sns.heatmap(cm_percent, annot=annotations, fmt='', cmap='Blues',
                    xticklabels=['Wake', 'Light', 'Deep', 'REM'],
                    yticklabels=['Wake', 'Light', 'Deep', 'REM'])

        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, f'confusion_matrix_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_training_curves(self, train_losses, val_losses, val_overall_kappas, val_median_kappas):
        """Plot training curves with median metrics"""
        epochs = range(1, len(train_losses) + 1)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Overall Kappa curve
        ax2.plot(epochs, val_overall_kappas, 'g-', label='Val Overall Kappa')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Kappa')
        ax2.set_title('Validation Overall Kappa')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Median Kappa curve
        ax3.plot(epochs, val_median_kappas, 'm-', label='Val Median Kappa')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Kappa')
        ax3.set_title('Validation Median Kappa')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Both Kappa curves
        ax4.plot(epochs, val_overall_kappas, 'g-', label='Overall Kappa')
        ax4.plot(epochs, val_median_kappas, 'm-', label='Median Kappa')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Kappa')
        ax4.set_title('Overall vs Median Kappa')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_curves.png'), dpi=300)
        plt.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Cross-Attention Sleep Model')
    parser.add_argument('--config', type=str, default='config_crossattn_v2.yaml',
                        help='Path to configuration file')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of runs')
    args = parser.parse_args()

    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'data': {
                'ppg_file': "../../data/mesa_ppg_with_labels.h5",
                'ecg_file': "../../data/mesa_real_ecg.h5",
                'index_file': "../../data/mesa_subject_index.h5",
                'num_workers': 4
            },
            'training': {
                'batch_size': 2,
                'num_epochs': 50,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'patience': 15
            },
            'model': {
                'd_model': 256,
                'n_heads': 8,
                'n_fusion_blocks': 3
            },
            'output': {
                'save_dir': "./outputs",
                'save_frequency': 5
            },
            'use_amp': True
        }

    # Multiple runs
    n_runs = args.runs
    all_results = []

    for run in range(1, n_runs + 1):
        print(f"\n{'=' * 80}")
        print(f"RUN {run}/{n_runs}")
        print('=' * 80)

        # Set random seeds
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + run)

        # Train
        trainer = CrossAttentionTrainer(config, run_id=run)
        results = trainer.train()
        all_results.append(results)

        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()

    # Summarize results
    print("\n" + "=" * 80)
    print(f"FINAL RESULTS ({n_runs} runs)")
    print("=" * 80)

    # Overall metrics
    overall_accuracies = [r['test_accuracy_overall'] for r in all_results]
    overall_kappas = [r['test_kappa_overall'] for r in all_results]
    overall_f1_scores = [r['test_f1_overall'] for r in all_results]

    # Median metrics
    median_accuracies = [r['test_accuracy_median'] for r in all_results]
    median_kappas = [r['test_kappa_median'] for r in all_results]
    median_f1_scores = [r['test_f1_median'] for r in all_results]

    print(f"\nCross-Attention PPG+ECG Model V2:")

    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {np.median(overall_accuracies):.4f} (median), "
          f"{np.mean(overall_accuracies):.4f}±{np.std(overall_accuracies):.4f} (mean±std)")
    print(f"  Kappa: {np.median(overall_kappas):.4f} (median), "
          f"{np.mean(overall_kappas):.4f}±{np.std(overall_kappas):.4f} (mean±std)")
    print(f"  F1 Score: {np.median(overall_f1_scores):.4f} (median), "
          f"{np.mean(overall_f1_scores):.4f}±{np.std(overall_f1_scores):.4f} (mean±std)")

    print(f"\nPer-Patient Median Metrics:")
    print(f"  Accuracy: {np.median(median_accuracies):.4f} (median), "
          f"{np.mean(median_accuracies):.4f}±{np.std(median_accuracies):.4f} (mean±std)")
    print(f"  Kappa: {np.median(median_kappas):.4f} (median), "
          f"{np.mean(median_kappas):.4f}±{np.std(median_kappas):.4f} (mean±std)")
    print(f"  F1 Score: {np.median(median_f1_scores):.4f} (median), "
          f"{np.mean(median_f1_scores):.4f}±{np.std(median_f1_scores):.4f} (mean±std)")

    print(f"\nAll overall kappas: {[f'{k:.4f}' for k in overall_kappas]}")
    print(f"All median kappas: {[f'{k:.4f}' for k in median_kappas]}")

    # Save summary results
    summary_results = {
        'num_runs': n_runs,
        'overall_metrics': {
            'accuracy': {
                'median': float(np.median(overall_accuracies)),
                'mean': float(np.mean(overall_accuracies)),
                'std': float(np.std(overall_accuracies)),
                'all': overall_accuracies
            },
            'kappa': {
                'median': float(np.median(overall_kappas)),
                'mean': float(np.mean(overall_kappas)),
                'std': float(np.std(overall_kappas)),
                'all': overall_kappas
            },
            'f1_score': {
                'median': float(np.median(overall_f1_scores)),
                'mean': float(np.mean(overall_f1_scores)),
                'std': float(np.std(overall_f1_scores)),
                'all': overall_f1_scores
            }
        },
        'per_patient_median_metrics': {
            'accuracy': {
                'median': float(np.median(median_accuracies)),
                'mean': float(np.mean(median_accuracies)),
                'std': float(np.std(median_accuracies)),
                'all': median_accuracies
            },
            'kappa': {
                'median': float(np.median(median_kappas)),
                'mean': float(np.mean(median_kappas)),
                'std': float(np.std(median_kappas)),
                'all': median_kappas
            },
            'f1_score': {
                'median': float(np.median(median_f1_scores)),
                'mean': float(np.mean(median_f1_scores)),
                'std': float(np.std(median_f1_scores)),
                'all': median_f1_scores
            }
        },
        'all_runs': all_results
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_path = os.path.join(config['output']['save_dir'],
                                f'crossattn_v2_summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=2)

    print(f"\nSummary results saved to: {summary_path}")


if __name__ == "__main__":
    main()