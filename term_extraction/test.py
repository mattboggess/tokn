import argparse
import torch
import json
from tqdm import tqdm
import model.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from model.metric import get_term_predictions, compute_term_categories
from parse_config import ConfigParser
from collections import Counter
import pickle

def main(config, split, out_dir, model_version):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data, split=split, shuffle=False)

    # build model architecture
    model = config.init_obj('arch', module_arch)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    sentence_metrics = [getattr(module_metric, met) for met in config['sentence_metrics']]
    term_metrics = [getattr(module_metric, met) for met in config['term_metrics']]

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

    total_loss = 0.0
    epoch_target = []
    epoch_pred = []
    epoch_sent_terms = []
    epoch_terms = []
    epoch_probs = []
    
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(data_loader)):
            
            batch_data['data'] = batch_data['data'].to(device)
            batch_data['target'] = batch_data['target'].to(device)
            batch_data['pad_mask'] = batch_data['pad_mask'].to(device)
            batch_data['bert_mask'] = batch_data['bert_mask'].to(device)
            
            if len(batch_data['target'].shape) < 2:
                batch_data['target'] = batch_data['target'].unsqueeze(0)
            
            output = model(batch_data)
            if config['arch']['type'] == 'BertCRFNER': 
                pred = model.decode(output, batch_data['bert_mask'])
            else:
                pred = torch.argmax(output, dim=-1)
                    
            loss = loss_fn(output, batch_data['target'], batch_data['bert_mask'],
                           data_loader.dataset.class_weights.to(device),
                           model)
            
            # get predicted terms with probabilities for each sentence
            terms, probs = get_term_predictions(output,  
                                                batch_data['bert_mask'], 
                                                batch_data['sentences'], 
                                                data_loader.tags)
            epoch_terms += terms
            epoch_probs += probs
            
            # get token level predictions
            epoch_pred += torch.flatten(pred * batch_data['bert_mask']).tolist()
            epoch_target += torch.flatten(batch_data['target'] * batch_data['bert_mask']).tolist()

            batch_size = batch_data["data"].shape[0]
            total_loss += loss.item() * batch_size

        
    # write out term predictions for each sentence with original data
    data = data_loader.dataset.data
    data['predicted_terms'] = epoch_terms
    data['term_probs'] = epoch_probs
    filename = f"{out_dir}/{split}-sentence-term-predictions.pkl"
    with open(filename, 'wb') as fid:
        pickle.dump(data, fid)

    # write out term classifications with confidence scores 
    term_classifications = compute_term_categories(data)
    filename = f"{out_dir}/{split}-term-classifications.json"
    with open(filename, 'w') as f:
        json.dump(term_classifications, f, indent=4)
        
    # write out metrics (both token and term level)
    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        m.__name__: m(epoch_target, epoch_pred) for m in sentence_metrics
    })
    log.update({
        m.__name__: m(term_classifications) for m in term_metrics
    })
    filename = f"{out_dir}/{split}-metrics.json"
    with open(filename, 'w') as f:
        json.dump(log, f, indent=4)
    
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
    out_dir = '/'.join(args.parse_args().resume.split('/')[:-1])
    model_version = args.parse_args().resume.split('/')[-1].replace('.pth', '')
    main(config, split, out_dir, model_version)
