import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
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

from multimodal_sleep_model import SleepPPGNet, MultiModalSleepNet
from multimodal_dataset_aligned import get_dataloaders


class MultiModalTrainer:
    def __init__(self, config, run_id=None):
        self.config = config
        self.run_id = run_id
        self.device = torch.device(f'cuda:{config["gpu"]["device_id"]}' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")


        self.setup_directories()


        self.writer = SummaryWriter(self.log_dir)


        with open(os.path.join(self.checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def setup_directories(self):

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{self.config['model_type']}_{timestamp}"
        if self.run_id is not None:
            model_name += f"_run{self.run_id}"

        self.output_dir = os.path.join(self.config['output']['save_dir'], model_name)
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        self.results_dir = os.path.join(self.output_dir, 'results')

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def create_model(self):

        if self.config['model_type'] == 'ppg_only':
            model = SleepPPGNet()
        elif self.config['model_type'] == 'ppg_with_noise':
            from ppg_with_noise_baseline import PPGWithNoiseBaseline
            model = PPGWithNoiseBaseline(
                noise_config=self.config.get('noise_config', None)
            )
        elif self.config['model_type'] == 'multimodal':
            model = MultiModalSleepNet(
                fusion_strategy=self.config.get('fusion_strategy', 'attention')
            )
        else:
            raise ValueError(f"Unknown model_type: {self.config['model_type']}")

        return model.to(self.device)

    def calculate_class_weights(self, train_dataset):

        print("\nCalculating class weights...")


        all_labels = []


        sample_size = min(len(train_dataset), 50)

        for idx in tqdm(range(sample_size), desc="Sampling labels"):

            data = train_dataset[idx]
            if len(data) == 2:  
                _, labels = data
            else:  
                _, _, labels = data


            valid_labels = labels[labels != -1]
            all_labels.extend(valid_labels.numpy().tolist())


        label_counts = Counter(all_labels)


        class_counts = []
        for i in range(4):
            class_counts.append(label_counts.get(i, 1))

        total_samples = sum(class_counts)

 
        print(f"\nLabel distribution (from {sample_size} subjects):")
        stage_names = ['Wake', 'Light', 'Deep', 'REM']
        for i, count in enumerate(class_counts):
            percentage = count / total_samples * 100
            print(f"  {stage_names[i]} (class {i}): {count} samples ({percentage:.2f}%)")


        class_weights = torch.tensor([1.0 / count for count in class_counts], dtype=torch.float32)


        class_weights = class_weights / class_weights.sum() * len(class_weights)

        print(f"\nClass weights: {class_weights}")

        return class_weights.to(self.device)

    def train_epoch(self, dataloader, model, device, optimizer, criterion):

        model.train()
        running_loss = 0.0
        total = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):

            if len(batch) == 2:  
                ppg, labels = batch
                ppg = ppg.to(device)
                labels = labels.to(device)
                outputs = model(ppg)
            else:  
                ppg, ecg, labels = batch
                ppg = ppg.to(device)
                ecg = ecg.to(device)
                labels = labels.to(device)
                outputs = model(ppg, ecg)

            optimizer.zero_grad()


            outputs = outputs.permute(0, 2, 1)  # (B, 1200, 4)
            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                labels.reshape(-1)
            ).mean()

            loss.backward()


            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()


            mask = labels != -1
            valid_labels = labels[mask]

            total += valid_labels.size(0)
            running_loss += loss.item() * valid_labels.size(0)


            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        gc.collect()
        torch.cuda.empty_cache()

        epoch_loss = running_loss / total if total > 0 else 0

        return epoch_loss

    def validate(self, dataloader, model, device, criterion):

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0


        all_preds = []
        all_labels = []


        patient_predictions = defaultdict(list)
        patient_labels = defaultdict(list)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):

                if len(batch) == 2:  
                    ppg, labels = batch
                    ppg = ppg.to(device)
                    labels = labels.to(device)
                    outputs = model(ppg)
                else:  
                    ppg, ecg, labels = batch
                    ppg = ppg.to(device)
                    ecg = ecg.to(device)
                    labels = labels.to(device)
                    outputs = model(ppg, ecg)

                outputs = outputs.permute(0, 2, 1)
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[-1]),
                    labels.reshape(-1)
                ).mean()


                batch_size = outputs.shape[0]
                for i in range(batch_size):
                    patient_idx = batch_idx * dataloader.batch_size + i


                    mask = labels[i] != -1
                    if mask.any():
                        patient_outputs = outputs[i][mask]
                        patient_labels_i = labels[i][mask]

                        _, predicted = patient_outputs.max(1)


                        patient_predictions[patient_idx].extend(predicted.cpu().numpy())
                        patient_labels[patient_idx].extend(patient_labels_i.cpu().numpy())

 
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(patient_labels_i.cpu().numpy())

                        correct += predicted.eq(patient_labels_i).sum().item()
                        total += patient_labels_i.numel()
                        running_loss += loss.item() * patient_labels_i.numel()

                gc.collect()
                torch.cuda.empty_cache()


        patient_accuracies = []
        patient_kappas = []
        patient_f1s = []

        for patient_idx in patient_predictions:
            if len(patient_predictions[patient_idx]) > 0:
                patient_acc = np.mean(np.array(patient_predictions[patient_idx]) ==
                                      np.array(patient_labels[patient_idx]))
                patient_accuracies.append(patient_acc)


                unique_labels = np.unique(patient_labels[patient_idx])
                if len(unique_labels) > 1:
                    patient_kappa = cohen_kappa_score(patient_labels[patient_idx],
                                                      patient_predictions[patient_idx])
                    patient_kappas.append(patient_kappa)

                patient_f1 = f1_score(patient_labels[patient_idx],
                                      patient_predictions[patient_idx],
                                      average='weighted')
                patient_f1s.append(patient_f1)


        epoch_loss = running_loss / len(all_labels) if all_labels else 0
        overall_accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) if all_labels else 0
        overall_kappa = cohen_kappa_score(all_labels, all_preds) if all_labels else 0
        overall_f1 = f1_score(all_labels, all_preds, average='weighted') if all_labels else 0


        median_accuracy = np.median(patient_accuracies) if patient_accuracies else 0
        median_kappa = np.median(patient_kappas) if patient_kappas else 0
        median_f1 = np.median(patient_f1s) if patient_f1s else 0

        # 打印per-patient kappa分布信息
        if patient_kappas:
            print(f"\nPer-patient Kappa distribution:")
            print(f"  Min: {np.min(patient_kappas):.4f}")
            print(f"  25%: {np.percentile(patient_kappas, 25):.4f}")
            print(f"  Median: {median_kappa:.4f}")
            print(f"  75%: {np.percentile(patient_kappas, 75):.4f}")
            print(f"  Max: {np.max(patient_kappas):.4f}")

        # 计算混淆矩阵（用overall数据）
        cm = confusion_matrix(all_labels, all_preds)
        per_class_metrics = self.calculate_per_class_metrics(cm)

        # 返回所有指标
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
        """计算每个类别的指标"""
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

    def plot_confusion_matrix(self, cm, epoch):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Wake', 'Light', 'Deep', 'REM'],
                    yticklabels=['Wake', 'Light', 'Deep', 'REM'])
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        save_path = os.path.join(self.results_dir, f'confusion_matrix_epoch_{epoch}.png')
        plt.savefig(save_path)
        plt.close()

    def train(self):
        """主训练流程"""
        print(f"\n{'=' * 60}")
        print(f"Training {self.config['model_type']} model")
        if self.run_id is not None:
            print(f"Run {self.run_id}/{self.config['training']['num_runs']}")
        print(f"{'=' * 60}")

        # 准备数据路径
        data_paths = {
            'ppg': self.config['data']['ppg_file'],
            'index': self.config['data']['index_file']
        }

        # 创建数据加载器
        print(f"\nLoading data...")

        if self.config['model_type'] in ['ppg_only', 'ppg_with_noise']:
            train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_dataloaders(
                data_paths,
                batch_size=self.config['data']['batch_size'],
                num_workers=self.config['data']['num_workers'],
                model_type='ppg_only',
                use_sleepppg_test_set=self.config['training']['use_sleepppg_test_set']
            )
        else:
            data_paths['real_ecg'] = self.config['data']['ecg_file']
            train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_dataloaders(
                data_paths,
                batch_size=self.config['data']['batch_size'],
                num_workers=self.config['data']['num_workers'],
                model_type='multimodal',
                use_generated_ecg=False,
                use_sleepppg_test_set=self.config['training']['use_sleepppg_test_set']
            )

        # 创建模型
        model = self.create_model()
        print(f"Model created on {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        # 计算类别权重
        class_weights = self.calculate_class_weights(train_dataset)

        # 损失函数和优化器
        train_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1, reduction="none")
        val_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.7, patience=5, verbose=True
        )

        # 训练设置
        best_validation_loss = float('inf')
        best_validation_kappa = 0
        best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')

        patience = self.config['training']['patience']
        trigger_times = 0

        training_history = {
            'train_losses': [],
            'val_losses': [],
            'val_overall_accuracies': [],
            'val_overall_kappas': [],
            'val_overall_f1_scores': [],
            'val_median_accuracies': [],
            'val_median_kappas': [],
            'val_median_f1_scores': []
        }

        # 训练循环
        num_epochs = self.config['training']['num_epochs']

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print('=' * 50)

            # 训练
            train_loss = self.train_epoch(
                train_loader, model, self.device, optimizer, train_criterion
            )
            training_history['train_losses'].append(train_loss)
            print(f"Training Loss: {train_loss:.4f}")

            # 验证
            val_results = self.validate(val_loader, model, self.device, val_criterion)

            # 保存历史
            training_history['val_losses'].append(val_results['loss'])
            training_history['val_overall_accuracies'].append(val_results['overall_accuracy'])
            training_history['val_overall_kappas'].append(val_results['overall_kappa'])
            training_history['val_overall_f1_scores'].append(val_results['overall_f1'])
            training_history['val_median_accuracies'].append(val_results['median_accuracy'])
            training_history['val_median_kappas'].append(val_results['median_kappa'])
            training_history['val_median_f1_scores'].append(val_results['median_f1'])

            print(f"\nValidation Results:")
            print(f"  Loss: {val_results['loss']:.4f}")
            print(f"  Overall - Acc: {val_results['overall_accuracy']:.4f}, "
                  f"Kappa: {val_results['overall_kappa']:.4f}, F1: {val_results['overall_f1']:.4f}")
            print(f"  Median  - Acc: {val_results['median_accuracy']:.4f}, "
                  f"Kappa: {val_results['median_kappa']:.4f}, F1: {val_results['median_f1']:.4f}")

            # 记录到tensorboard
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Val/Loss', val_results['loss'], epoch)
            self.writer.add_scalar('Val/Overall_Accuracy', val_results['overall_accuracy'], epoch)
            self.writer.add_scalar('Val/Overall_Kappa', val_results['overall_kappa'], epoch)
            self.writer.add_scalar('Val/Overall_F1', val_results['overall_f1'], epoch)
            self.writer.add_scalar('Val/Median_Accuracy', val_results['median_accuracy'], epoch)
            self.writer.add_scalar('Val/Median_Kappa', val_results['median_kappa'], epoch)
            self.writer.add_scalar('Val/Median_F1', val_results['median_f1'], epoch)

            # 学习率调整（使用overall kappa）
            scheduler.step(val_results['overall_kappa'])

            # 保存最佳模型（基于overall kappa）
            if val_results['overall_kappa'] > best_validation_kappa:
                best_validation_kappa = val_results['overall_kappa']

            if val_results['loss'] < best_validation_loss:
                best_validation_loss = val_results['loss']
                trigger_times = 0

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_validation_loss': best_validation_loss,
                    'best_overall_kappa': val_results['overall_kappa'],
                    'best_median_kappa': val_results['median_kappa'],
                    'training_history': training_history,
                    'config': self.config
                }
                torch.save(checkpoint, best_model_path)
                print('Saved best model!')

                # 绘制混淆矩阵
                if self.config['output']['save_intermediate']:
                    self.plot_confusion_matrix(val_results['confusion_matrix'], epoch)
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}!")
                    break

            # 定期保存检查点
            if epoch % self.config['output']['save_frequency'] == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training_history': training_history
                }, checkpoint_path)

        print("\nTraining completed!")

        # 测试最佳模型
        print("\n" + "=" * 60)
        print("Testing best model on SleepPPG-Net test set...")
        print("=" * 60)

        # 加载最佳模型
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        # 测试
        test_results = self.validate(test_loader, model, self.device, val_criterion)

        print(f"\nTest Results:")
        print(f"  Loss: {test_results['loss']:.4f}")
        print(f"  Overall - Acc: {test_results['overall_accuracy']:.4f}, "
              f"Kappa: {test_results['overall_kappa']:.4f}, F1: {test_results['overall_f1']:.4f}")
        print(f"  Median  - Acc: {test_results['median_accuracy']:.4f}, "
              f"Kappa: {test_results['median_kappa']:.4f}, F1: {test_results['median_f1']:.4f}")

        # 详细分类报告
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

        # 绘制最终混淆矩阵
        self.plot_confusion_matrix(test_results['confusion_matrix'], 'final')

        # 保存结果
        results = {
            'model_type': self.config['model_type'],
            'run_id': self.run_id,
            'test_loss': test_results['loss'],
            'test_accuracy_overall': test_results['overall_accuracy'],
            'test_kappa_overall': test_results['overall_kappa'],
            'test_f1_overall': test_results['overall_f1'],
            'test_accuracy_median': test_results['median_accuracy'],
            'test_kappa_median': test_results['median_kappa'],
            'test_f1_median': test_results['median_f1'],
            'classification_report': report,
            'confusion_matrix': test_results['confusion_matrix'].tolist(),
            'training_history': training_history,
            'best_epoch': checkpoint['epoch'],
            'patient_kappa_stats': {
                'min': float(np.min(test_results['patient_kappas'])),
                'max': float(np.max(test_results['patient_kappas'])),
                'mean': float(np.mean(test_results['patient_kappas'])),
                'std': float(np.std(test_results['patient_kappas'])),
                'median': float(test_results['median_kappa']),
                '25_percentile': float(np.percentile(test_results['patient_kappas'], 25)),
                '75_percentile': float(np.percentile(test_results['patient_kappas'], 75))
            }
        }

        results_path = os.path.join(self.results_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_path}")

        self.writer.close()

        return results


def parse_args():
    parser = argparse.ArgumentParser(description='Train sleep staging models')
    parser.add_argument('--config', type=str, default='config_cloud.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, choices=['ppg_only', 'ppg_with_noise', 'real_ecg', 'both'],
                        default='both', help='Which model(s) to train')
    parser.add_argument('--runs', type=int, default=None,
                        help='Number of runs (overrides config)')
    return parser.parse_args()


def main():
    """主函数 - 支持多次运行"""
    args = parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 命令行参数覆盖
    if args.runs is not None:
        config['training']['num_runs'] = args.runs

    num_runs = config['training'].get('num_runs', 1)

    # 存储所有运行的结果
    all_results = defaultdict(list)

    # 决定要训练的模型
    models_to_train = []
    if args.model == 'both':
        if config.get('model', {}).get('ppg_only', {}).get('enabled', False):
            models_to_train.append('ppg_only')
        if config.get('model', {}).get('ppg_with_noise', {}).get('enabled', False):
            models_to_train.append('ppg_with_noise')
        if config.get('model', {}).get('real_ecg', {}).get('enabled', False):
            models_to_train.append('real_ecg')
    else:
        models_to_train = [args.model]

    # 多次运行
    for run in range(1, num_runs + 1):
        print(f"\n{'=' * 80}")
        print(f"RUN {run}/{num_runs}")
        print('=' * 80)

        # 设置随机种子
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + run)

        # 训练每个模型
        for model_type in models_to_train:
            print(f"\n{'=' * 60}")
            print(f"Training {model_type} model")
            print('=' * 60)

            # 准备配置
            model_config = config.copy()
            model_config['model_type'] = model_type

            if model_type == 'real_ecg':
                model_config['fusion_strategy'] = config['model']['real_ecg']['fusion_strategy']

            # 创建训练器并训练
            trainer = MultiModalTrainer(model_config, run_id=run if num_runs > 1 else None)
            results = trainer.train()

            # 保存结果
            all_results[model_type].append(results)

            # 清理GPU内存
            torch.cuda.empty_cache()
            gc.collect()

    # 计算并打印最终统计结果
    if num_runs > 1:
        print("\n" + "=" * 80)
        print(f"FINAL RESULTS ({num_runs} runs)")
        print("=" * 80)

        for model_type in models_to_train:
            if all_results[model_type]:
                # Overall指标
                overall_accuracies = [r['test_accuracy_overall'] for r in all_results[model_type]]
                overall_kappas = [r['test_kappa_overall'] for r in all_results[model_type]]
                overall_f1_scores = [r['test_f1_overall'] for r in all_results[model_type]]

                # Median指标
                median_accuracies = [r['test_accuracy_median'] for r in all_results[model_type]]
                median_kappas = [r['test_kappa_median'] for r in all_results[model_type]]
                median_f1_scores = [r['test_f1_median'] for r in all_results[model_type]]

                print(f"\n{model_type.upper()} Model:")
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

        # 保存汇总结果
        summary_results = {
            'num_runs': num_runs,
            'results': {}
        }

        for model_type in models_to_train:
            if all_results[model_type]:
                summary_results['results'][model_type] = {
                    'all_runs': all_results[model_type],
                    'overall_metrics': {
                        'accuracy': {
                            'median': float(np.median([r['test_accuracy_overall'] for r in all_results[model_type]])),
                            'mean': float(np.mean([r['test_accuracy_overall'] for r in all_results[model_type]])),
                            'std': float(np.std([r['test_accuracy_overall'] for r in all_results[model_type]])),
                            'all': [r['test_accuracy_overall'] for r in all_results[model_type]]
                        },
                        'kappa': {
                            'median': float(np.median([r['test_kappa_overall'] for r in all_results[model_type]])),
                            'mean': float(np.mean([r['test_kappa_overall'] for r in all_results[model_type]])),
                            'std': float(np.std([r['test_kappa_overall'] for r in all_results[model_type]])),
                            'all': [r['test_kappa_overall'] for r in all_results[model_type]]
                        },
                        'f1_score': {
                            'median': float(np.median([r['test_f1_overall'] for r in all_results[model_type]])),
                            'mean': float(np.mean([r['test_f1_overall'] for r in all_results[model_type]])),
                            'std': float(np.std([r['test_f1_overall'] for r in all_results[model_type]])),
                            'all': [r['test_f1_overall'] for r in all_results[model_type]]
                        }
                    },
                    'per_patient_median_metrics': {
                        'accuracy': {
                            'median': float(np.median([r['test_accuracy_median'] for r in all_results[model_type]])),
                            'mean': float(np.mean([r['test_accuracy_median'] for r in all_results[model_type]])),
                            'std': float(np.std([r['test_accuracy_median'] for r in all_results[model_type]])),
                            'all': [r['test_accuracy_median'] for r in all_results[model_type]]
                        },
                        'kappa': {
                            'median': float(np.median([r['test_kappa_median'] for r in all_results[model_type]])),
                            'mean': float(np.mean([r['test_kappa_median'] for r in all_results[model_type]])),
                            'std': float(np.std([r['test_kappa_median'] for r in all_results[model_type]])),
                            'all': [r['test_kappa_median'] for r in all_results[model_type]]
                        },
                        'f1_score': {
                            'median': float(np.median([r['test_f1_median'] for r in all_results[model_type]])),
                            'mean': float(np.mean([r['test_f1_median'] for r in all_results[model_type]])),
                            'std': float(np.std([r['test_f1_median'] for r in all_results[model_type]])),
                            'all': [r['test_f1_median'] for r in all_results[model_type]]
                        }
                    }
                }

        # 保存汇总结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = os.path.join(config['output']['save_dir'], f'summary_results_{timestamp}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_results, f, indent=2)

        print(f"\nSummary results saved to: {summary_path}")

    print("\n" + "=" * 80)
    print("ALL TRAINING COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
