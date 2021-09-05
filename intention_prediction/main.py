import os
import sys
import time
import copy
import logging
import argparse
import numpy as np
import random
from pathlib import Path
from sklearn.metrics import confusion_matrix

from data import build_dataset, VIDEO_FPS
from model import IntentionFFNN, IntentionRNN, IntentionTransformer

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def get_args_parser():
    parser = argparse.ArgumentParser('Intention Prediction', add_help=False)
    
    parser.add_argument('--exp_name', default='intention', type=str,
                        help="Identifier for the current experiment")

    # Training conditions
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr_drop', default=5, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--model_type', default='ffnn', type=str,
                        help="Type of model to train ['ffnn', 'lstm', 'gru', 'transformer']")
    parser.add_argument('--hidden_size', default=128, type=int,
                        help="Size of the hidden size")
    parser.add_argument('--num_layers', default=2, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--num_output', default=3, type=int,
                        help="Number of future intention predictions (setting for multi-task learning)")
    parser.add_argument('--num_heads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    # dataset parameters
    parser.add_argument('--data_path', default='../data_csv', type=str,
                        help="")
    parser.add_argument('--output_dir', default='./result', type=str,
                        help='path where to save, empty for no saving')
    parser.add_argument('--context_length', default=1.0, type=float,
                        help="Context length in seconds")
    parser.add_argument('--state_info', action="store_false",
                        help="State information as a feature")
    parser.add_argument("--future_stamps", default=[0.5, 1, 1.5], nargs="+", type=float,
                        help="Future time stamps to predict")
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=2, type=int)

    return parser


def main(args):
    output_dir = Path(args.output_dir) / args.exp_name
    logging.basicConfig(level=logging.DEBUG, filename=output_dir/"logging.log")
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    
    logging.info(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.model_type == 'ffnn':
        model = IntentionFFNN(
            input_size=int(args.context_length * VIDEO_FPS / 2) * (15 if args.state_info else 14),
            hidden_size=args.hidden_size,
            num_output=args.num_output,
            num_layers=args.num_layers)
    elif args.model_type == 'lstm':
        model = IntentionRNN(
            input_size=15 if args.state_info else 14,
            hidden_size=args.hidden_size,
            num_output=args.num_output,
            num_layers=args.num_layers,
            rnn_type=args.model_type)
    elif args.model_type == 'gru':
        model = IntentionRNN(
            input_size=15 if args.state_info else 14,
            hidden_size=args.hidden_size,
            num_output=args.num_output,
            num_layers=args.num_layers,
            rnn_type=args.model_type)
    else:
        model = IntentionTransformer(
            input_size=16,
            hidden_size=args.hidden_size,
            num_output=args.num_output,
            num_layers=args.num_layers,
            num_heads=args.num_heads)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('number of params: {}'.format(n_parameters))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val   = build_dataset(image_set='val',   args=args)
    dataset_test  = build_dataset(image_set='test',  args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val   = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test  = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   num_workers=args.num_workers)
    data_loader_val   = DataLoader(dataset_val, 1, sampler=sampler_val,
                                   drop_last=False, num_workers=args.num_workers)
    data_loader_test  = DataLoader(dataset_test, 1, sampler=sampler_test,
                                   drop_last=False, num_workers=args.num_workers)   
    logging.info("dataset built!")
    logging.info("Dataset Lengths:\n\t{:>5}: {:4d} instances\n\t{:>5}: {:4d} instances\n\t{:>5}: {:4d} instances".format("train", len(dataset_train), "val", len(dataset_val), "test", len(dataset_test)))

    criterion = nn.CrossEntropyLoss()

    logging.info("Start training")
    start_time = time.time()
    best_acc = 0.0
    for epoch in range(args.epochs):
        logging.info('-' * 25)
        logging.info('Epoch {}/{}'.format(epoch, args.epochs - 1))
        logging.info('-' * 25)
        model.train()
        running_loss, running_corrects = 0.0, [0 for _ in range(args.num_output)]

        for idx, (datum, label) in enumerate(data_loader_train):
            datum = datum.to(device)
            label = label.to(device)

            outputs = model(datum.float())
            preds = [torch.max(o, 1) for o in outputs]
            preds = [p[1] for p in preds]
            loss = sum([criterion(outputs[i], label[:, i].long()) for i in range(args.num_output)])
            
            optimizer.zero_grad()
            loss.backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            optimizer.step()

            # statistics
            running_loss += loss.item() * datum.size(0)
            for i in range(args.num_output):
                running_corrects[i] += torch.sum(preds[i] == label[:, i].data)

        lr_scheduler.step()

        epoch_loss = running_loss / len(data_loader_train)
        epoch_acc = [running_corrects[i].double() / (len(data_loader_train) * args.batch_size) for i in range(args.num_output)]
        train_report = "-Train\n\tLoss: {:.4f}".format(epoch_loss)
        for i in range(args.num_output):
            train_report += " Accuracy[{}sec]: {:.4f}".format(args.future_stamps[i], epoch_acc[i])
        logging.info(train_report)

        val_acc, val_label, val_pred = evaluate(args, model, data_loader_val, device)
        val_report = "-Validation\n\tLoss: {:.4f}".format(0.0)
        for i in range(args.num_output):
            val_report += " Accuracy[{}sec]: {:.4f}".format(args.future_stamps[i], val_acc[i])
        logging.info(val_report)

        if best_acc <= sum(val_acc) / 3:
            best_acc = sum(val_acc) / 3
            best_model = copy.deepcopy(model.state_dict())
            if args.output_dir:
                checkpoint_path = output_dir / f'checkpoint{epoch:04}.pth'
                torch.save(model.state_dict(), checkpoint_path)

    total_time = time.time() - start_time
    logging.info('Training complete in {:.0f}m {:.0f}s ({:.4f})'.format(
        total_time // 60, total_time % 60, total_time))
    
    model.load_state_dict(best_model)
    test_acc, test_label, test_pred = evaluate(args, model, data_loader_test, device)
    total_tn, total_fp, total_fn, total_tp = [], [], [], []
    test_report_accuracy = "-Test\n\tLoss: {:.4f}".format(0.0)
    test_report_tptnfpfn = "\n"
    for i in range(args.num_output):
        test_report_accuracy += " Accuracy[{}sec]: {:.4f}".format(args.future_stamps[i], test_acc[i])
        tn, fp, fn, tp = confusion_matrix(test_label[i], test_pred[i]).ravel()
        test_report_tptnfpfn += "\t[{:.2f}sec] ACC: {:.4f}, F1: {:.4f}, TP: {}, TN: {}, FP: {}, FN: {}\n".format(args.future_stamps[i], (tp+tn)/(tp+tn+fp+fn), (2*tp)/(2*tp+fp+fn), tp, tn, fp, fn)
    logging.info(test_report_accuracy+test_report_tptnfpfn)


def evaluate(args, model, data_loader, device):
    running_corrects = [0 for _ in range(args.num_output)]
    total_preds, total_labels = [[] for _ in range(args.num_output)], [[] for _ in range(args.num_output)]
    
    model.eval()
    for datum, label in data_loader:
        datum = datum.to(device)
        label = label.to(device)
        
        with torch.set_grad_enabled(False):
            outputs = model(datum.float())
            preds = [torch.max(o, 1) for o in outputs]
            preds = [p[1] for p in preds]
            for i in range(args.num_output):
                total_labels[i].extend(label[:, i].tolist())
                total_preds[i].extend(preds[i].tolist())
                running_corrects[i] += torch.sum(preds[i] == label[:, i].data)
    acc = [running_corrects[i].double() / len(data_loader) for i in range(args.num_output)]
    return acc, total_labels, total_preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Intention prediction training and evaluation script',
        parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        (Path(args.output_dir) / args.exp_name).mkdir(parents=True, exist_ok=True)
    main(args)
