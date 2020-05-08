import argparse
import torch
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
        
        epoch_data = {
            'text': [],
            'term_pair': [],
            'label': [],
            'relation': [],
            'prediction': [],
            'score': []
        }
        epoch_loss = []
        epoch_score = []
        
        for i, batch_data in enumerate(tqdm(data_loader)):
            
            for field in ['input_ids', 'label', 'target', 'attention_mask', 'term1_mask', 'term2_mask']:
                batch_data[field] = batch_data[field].to(device)

            output = model(batch_data)
            pred = torch.argmax(output, dim=-1)
            loss = loss_fn(output, batch_data['target'],
                           data_loader.dataset.class_weights.to(device))

            # accumulate epoch quantities 
            tmp = output.cpu().detach().numpy()
            epoch_score += [tmp]
            epoch_data['label'] += [t.item() for t in batch_data['label']]
            epoch_data['score'] += [tmp[i, :].tolist() for i in range(tmp.shape[0])]
            epoch_data['prediction'] += [p.item() for p in pred]
            epoch_data['term_pair'] += batch_data['term_pair']
            epoch_data['text'] += batch_data['text']
            epoch_data['relation'] += batch_data['relation']
            epoch_loss += [loss.item()]

    log = {'loss': np.sum(epoch_loss) / len(data_loader)}
    log.update({
        m.__name__: m(epoch_data['label'], epoch_score) for m in metric_fns
    })
    logger.info(log)
    
    # save out metrics across every single training instance
    filename = f"{out_dir}/{split}-instance_level-metrics.json"
    with open(filename, 'w') as f:
        json.dump(log, f, indent=4)
        
    # save out predictions for every instance 
    predictions = pd.DataFrame(epoch_data) 
    filename = f"{out_dir}/{split}-{model_version}-predictions.pkl"
    predictions.to_pickle(filename)


    # save out term pair level classifications 
    term_pair_class = get_term_classifications(predictions)
    filename = f"{out_dir}/{split}-term_pair-classifications.json"
    with open(filename, 'w') as f:
        json.dump(term_pair_class, f, indent=4)
        
    # save out term pair level metrics 
    term_metrics = compute_term_metrics(term_pair_class)
    filename = f"{out_dir}/{split}-term_pair-metrics.json"
    with open(filename, 'w') as f:
        json.dump(term_metrics, f, indent=4)
    print(log)
    print(term_metrics)


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
