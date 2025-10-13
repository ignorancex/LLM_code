import argparse
import os
import pandas as pd
import torch
import tqdm
import yaml

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification
)

from datasets import (
    AIBODataset,
    MSPDataset
)
from utils import (
    evaluate,
    create_aibo,
    create_data,
)

from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BERT Modelling')
    parser.add_argument(
        '--results-root', 
        required=True,
        help='Path to store results in'
    )
    parser.add_argument(
        '--labels', 
        required=True,
        help='Path to dataset labels'
    )
    parser.add_argument(
        '--transcriptions', 
        required=True,
        help='Path to transcriptions'
    )
    parser.add_argument(
        '--batch-size',
        default=8,
        type=int
    )
    parser.add_argument(
        '--learning-rate',
        default=1e-5,
        type=float
    )
    parser.add_argument(
        '--epochs',
        default=10,
        type=int
    )
    parser.add_argument(
        '--device',
        default='cpu'
    )
    parser.add_argument(
        '--dataset',
        default='msp',
        choices=['msp', 'aibo']
    )
    parser.add_argument(
        '--base-model', 
        default='bert-base-cased',
        choices=[
            'bert-base-uncased',
            'bert-base-cased',
            'FacebookAI/roberta-base',
            "distilbert/distilbert-base-uncased",
            "google/electra-base-discriminator",
            "dbmdz/bert-base-german-uncased",
            "T-Systems-onsite/german-roberta-sentence-transformer-v2",
            "distilbert/distilbert-base-german-cased",
            "german-nlp-group/electra-base-german-uncased"
        ]
    )
    args = parser.parse_args()

    experiment_folder = args.results_root
    os.makedirs(experiment_folder, exist_ok=True)
    writer = SummaryWriter(
        log_dir=os.path.join(experiment_folder, 'log')
    )
    
    if args.dataset == 'msp':
        df_train, df_dev, df_test, encoder = create_data(labels=args.labels)
        train_dataset = MSPDataset(
            df=df_train,
            path=args.transcriptions,
            target_transform=encoder
        )
        x, y = train_dataset[0]
        print(f"Dry run: {x}--[{y}]")

        dev_dataset = MSPDataset(
            df=df_dev,
            path=args.transcriptions,
            target_transform=encoder
        )
        test_dataset = MSPDataset(
            df=df_test,
            path=args.transcriptions,
            target_transform=encoder
        )
        target = "EmoClass"
    elif args.dataset == 'aibo':
        df_train, df_dev, df_test, encoder = create_aibo(labels=args.labels)
        target = "class"
        with open(args.transcriptions, "r") as fp:
            lines = fp.readlines()
        transliteration = pd.DataFrame.from_dict({x.split(" ")[0]: " ".join(x.split(" ")[1:-1]) for x in lines}, orient="index").reset_index()
        transliteration = transliteration.rename(columns={"index": "file", 0: "text"})
        transliteration = transliteration.set_index("file")
        
        train_dataset = AIBODataset(
            df=df_train,
            transliteration=transliteration,
            target_transform=encoder
        )
        x, y = train_dataset[0]

        dev_dataset = AIBODataset(
            df=df_dev,
            transliteration=transliteration,
            target_transform=encoder
        )
        test_dataset = AIBODataset(
            df=df_test,
            transliteration=transliteration,
            target_transform=encoder
        )
    else:
        raise NotImplementedError(args.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=2,
        drop_last=True
    )

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=2
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, 
        num_labels=len(encoder.labels)
    )
    evaluation_metric = 'UAR'
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.00001,
    )

    max_metric = -1e31
    best_epoch = 0
    best_state = None
    best_results = None
    frequency = (
        df_train[target]
        .map(encoder)
        .value_counts()
        .sort_index()
        .values
    )
    weight = torch.tensor(1 / frequency, dtype=torch.float32)
    weight /= weight.sum()
    criterion = torch.nn.CrossEntropyLoss(weight)

    if not os.path.exists(os.path.join(experiment_folder, 'state.pth.tar')):

        for epoch in range(args.epochs):
            model.to(args.device)
            model.train()
            for index, (features, targets) in tqdm.tqdm(
                enumerate(train_loader),
                desc=f'Epoch {epoch}',
                total=len(train_loader)
            ):
                features = tokenizer.batch_encode_plus(
                    list(features),
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                ).to(args.device)
                outputs = model(**features).logits.cpu()
                loss = criterion(outputs, targets)
                if index % 50 == 0:
                    writer.add_scalar(
                        'Loss',
                        loss,
                        epoch * len(train_loader) + index
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            dev_results, targets, outputs = evaluate(
                model=model, 
                device=args.device, 
                loader=dev_loader, 
                tokenizer=tokenizer,
                encoder=encoder
            )
            print(
                f'Dev results at epoch {epoch+1}:\n{yaml.dump(dev_results)}')
            for key in dev_results.keys():
                writer.add_scalar(f'Dev/{key}', dev_results[key], epoch)
            if dev_results[evaluation_metric] > max_metric:
                max_metric = dev_results[evaluation_metric]
                best_epoch = epoch
                best_state = model.cpu().state_dict()
                best_results = dev_results.copy()
        writer.close()
        print(
            f'Best dev results found at epoch {best_epoch+1}:\n{yaml.dump(best_results)}')
        best_results['Epoch'] = best_epoch + 1
        with open(os.path.join(experiment_folder, 'dev.yaml'), 'w') as fp:
            yaml.dump(best_results, fp)
    else:
        best_state = torch.load(os.path.join(
            experiment_folder, 'state.pth.tar'))
    torch.save(best_state, os.path.join(
        experiment_folder, 'state.pth.tar'))
    model.load_state_dict(best_state)
    test_results, targets, outputs = evaluate(
        model=model, 
        device=args.device, 
        loader=test_loader, 
        tokenizer=tokenizer,
        encoder=encoder
    )
    print(f'Test results:\n{yaml.dump(test_results)}')
    with open(os.path.join(experiment_folder, 'test.yaml'), 'w') as fp:
        yaml.dump(test_results, fp)
    df_test['prediction'] = outputs
    df_test.reset_index().to_csv(os.path.join(experiment_folder, 'results.csv'), index=False)