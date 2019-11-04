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


def main(config, split, out_dir):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data, split=split)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    sentence_metrics = [getattr(module_metric, met) for met in config['sentence_metrics']]
    term_metrics = [getattr(module_metric, met) for met in config['term_metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0

    epoch_target = []
    epoch_pred = []
    epoch_terms = Counter() 
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            batch_data["data"] = batch_data["data"].to(device)
            batch_data["target"] = batch_data["target"].to(device)
            batch_data["pad_mask"] = batch_data["pad_mask"].to(device)
            batch_data["bert_mask"] = batch_data["bert_mask"].to(device)
            
            output = model(batch_data)
            pred = torch.argmax(output, dim=-1)
            term_predictions = get_term_predictions(pred, batch_data["target"], 
                                                    batch_data["bert_mask"], 
                                                    batch_data["sentences"], data_loader.tags)
            epoch_target += term_predictions["target"]
            epoch_pred += term_predictions["prediction"]
            epoch_terms.update(term_predictions["predicted_terms"])

            loss = loss_fn(output, batch_data["target"], batch_data["bert_mask"])
            batch_size = batch_data["data"].shape[0]
            total_loss += loss.item() * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        m.__name__: m(epoch_target, epoch_pred) for m in sentence_metrics
    })
        
    # compute overall term identification metrics
    term_classifications = compute_term_categories(data_loader.dataset.term_counts,
                                                   epoch_terms)
    log.update(**{m.__name__: m(term_classifications) for m in term_metrics})
    
    
    filename = f"{out_dir}/classifications-test_eval-{split}.json"
    with open(filename, "w") as f:
        json.dump(term_classifications, f, indent=4)
    
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-o', '--output_dir', default=".", type=str,
                      help='output directory to place predictions')
    args.add_argument('-i', '--input_file', default=None, type=str,
                      help='input file containing text to make predictions on')

    config = ConfigParser.from_args(args)
    args = args.parse_args()
    
    main(config, args.input_file, args.output_dir)
