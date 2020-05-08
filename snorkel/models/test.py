import argparse
import torch
import numpy as np
import json
from tqdm import tqdm
import model.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
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
        
        epoch_target = []
        epoch_scores = []
        epoch_pred = []
        epoch_word_pairs = []
        epoch_loss = [] 
        
        for i, batch_data in enumerate(tqdm(data_loader)):
            
            for field in ['input_ids', 'label', 'target', 'attention_mask', 'term1_mask', 'term2_mask']:
                batch_data[field] = batch_data[field].to(device)

            output = model(batch_data)
            pred = torch.argmax(output, dim=-1)
            loss = loss_fn(output, batch_data['target'],
                           data_loader.dataset.class_weights.to(device))

            # accumulate epoch quantities 
            epoch_target += [t.item() for t in batch_data['label']]
            epoch_scores += [output.cpu().detach().numpy()]
            epoch_pred += [p.item() for p in pred]
            epoch_word_pairs += batch_data['term_pair']
            epoch_loss += [loss.item()]

    log = {'loss': np.sum(epoch_loss) / len(data_loader)}
    log.update({
        m.__name__: m(epoch_target, epoch_scores) for m in metric_fns
    })
    logger.info(log)
    
    filename = f"{out_dir}/{split}-{model_version}-metrics.json"
    with open(filename, 'w') as f:
        json.dump(log, f, indent=4)
        
    filename = f"{out_dir}/{split}-{model_version}-predictions.txt"
    with open(filename, 'w') as f:
        f.write('\n'.join([str(p) for p in epoch_pred]))
        
    print(log)


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