import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import model.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from model.metric import *
import model.model as module_arch
from parse_config import ConfigParser
from sklearn.metrics import classification_report


def main(config, split, out_dir, model_version):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data, split=split, shuffle=False)

    # build model architecture
    model = config.init_obj('arch', module_arch,
                            num_classes=data_loader.dataset.num_classes,
                            vocab_size=len(data_loader.dataset.tokenizer))

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        
        epoch_loss = []
        epoch_pred = []
        epoch_label = []
        epoch_score = []
        relation_classes = data_loader.dataset.relation_classes
        
        for i, batch_data in enumerate(tqdm(data_loader)):
            
            for field in ['input_ids', 'label', 'target', 'attention_mask', 'term1_mask', 'term2_mask']:
                batch_data[field] = batch_data[field].to(device)

            output = model(batch_data)
            pred = torch.argmax(output, dim=-1)
            loss = loss_fn(output, batch_data['target'],
                           data_loader.dataset.class_weights.to(device))

            # accumulate epoch quantities 
            epoch_score += list(F.softmax(output, dim=-1).cpu().detach().numpy().max(axis=-1))
            epoch_pred += [relation_classes[p.item()] for p in pred]
            epoch_label += [relation_classes[l.item()] for l in batch_data['label']]
            epoch_loss += [loss.item()]

    # write out instance level metrics
    print("Instance Metrics:")
    instance_metrics = classification_report(epoch_label, epoch_pred, 
                                             labels=[r for r in relation_classes if r != 'OTHER'])
    print(instance_metrics)
    with open(f"{out_dir}/{split}-instance-metrics.txt", 'w') as fid:
        fid.write(instance_metrics)
    
    # save out predictions for every instance 
    data = data_loader.dataset.data
    data['predicted_relation'] = epoch_pred
    data['prediction_confidence'] = [np.max(s) for s in epoch_score]
    filename = f"{out_dir}/{split}-sentence-predictions.pkl"
    data.to_pickle(filename)
    
    # get term pair classifications
    tp_preds = get_term_pair_predictions(data)
    filename = f"{out_dir}/{split}-term-pair-predictions.csv"
    tp_preds.sort_values('term_pair').to_csv(filename, index=False)
    
    # write out term pair metrics
    print("Term Pair Metrics:")
    tp_metrics = classification_report(tp_preds.true_label, tp_preds.predicted_label, 
                                       labels=[r for r in relation_classes if r != 'OTHER'])
    print(tp_metrics)
    kb_metrics = classification_report(tp_preds.true_label, tp_preds.kb_label, 
                                       labels=[r for r in relation_classes if r != 'OTHER'])
    lf_metrics = classification_report(tp_preds.true_label, tp_preds.label_fn_label, 
                                       labels=[r for r in relation_classes if r != 'OTHER'])

    # save out term pair level metrics 
    filename = f"{out_dir}/{split}-term-pair-metrics.txt"
    with open(filename, 'w') as fid:
        fid.write('Predicted Term Pair Metrics:\n' + tp_metrics + '\n' + \
                  'Bio101 Term Pair Metrics:\n' + kb_metrics + '\n' + \
                  'Label Fn Term Pair Metrics:\n' + lf_metrics)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--split', default=None, type=str,
                      help='data split you want to evaluate trained model on (default: None)')

    config = ConfigParser.from_args(args, test=True)
    split = args.parse_args().split
    out_dir = "/".join(args.parse_args().resume.split("/")[:-1])
    model_version = args.parse_args().resume.split("/")[-1].replace(".pth", "")
    main(config, split, out_dir, model_version)
