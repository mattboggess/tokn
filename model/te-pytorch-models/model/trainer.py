import numpy as np
import torch
import json
from utils import inf_loop, MetricTracker
from torchvision.utils import make_grid
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from model.metric import get_term_predictions, compute_term_categories
from collections import Counter

class Trainer:
    """
    Implements training and validation logic
    """
    def __init__(self, model, criterion, sentence_metric_ftns, term_metric_ftns, optimizer, config, 
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.sentence_metric_ftns = sentence_metric_ftns
        self.term_metric_ftns = term_metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)
            
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        
        epoch_target = []
        epoch_pred = []
        epoch_terms = Counter() 
        epoch_loss = 0
        
        for batch_idx, batch_data in enumerate(self.data_loader):
            
            batch_data["data"] = batch_data["data"].to(self.device)
            batch_data["target"] = batch_data["target"].to(self.device)
            batch_data["pad_mask"] = batch_data["pad_mask"].to(self.device)
            batch_data["bert_mask"] = batch_data["bert_mask"].to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(batch_data)
            with torch.no_grad():
                pred = torch.argmax(output, dim=-1)
            loss = self.criterion(output, batch_data["target"], batch_data["bert_mask"])
            loss.backward()
            self.optimizer.step()

            # compute term sentence level metrics
            term_predictions = get_term_predictions(pred, batch_data["target"], 
                                                    batch_data["bert_mask"], 
                                                    batch_data["sentences"], self.data_loader.tags)
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.writer.add_scalar("loss", loss.item())
            for met in self.sentence_metric_ftns:
                self.writer.add_scalar(met.__name__, met(term_predictions["target"], 
                                                         term_predictions["prediction"]))
                
            # update full epoch trackers
            epoch_target += term_predictions["target"]
            epoch_pred += term_predictions["prediction"]
            epoch_loss += loss.item()
            epoch_terms.update(term_predictions["predicted_terms"])
            
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
                
        # compute epoch level sentence metrics
        log = {m.__name__: m(epoch_target, epoch_pred) for m in self.sentence_metric_ftns}
        
        # compute overall term identification metrics
        term_classifications = compute_term_categories(self.data_loader.dataset.term_counts,
                                                       epoch_terms)
        log.update(**{m.__name__: m(term_classifications) for m in self.term_metric_ftns})
        
        log["loss"] = epoch_loss / batch_idx

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        #if self.lr_scheduler is not None:
        #    self.lr_scheduler.step()
        
        self.train_classifications = term_classifications
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        with torch.no_grad():
            
            epoch_target = []
            epoch_pred = []
            epoch_terms = Counter() 
            epoch_loss = 0
            
            for batch_idx, batch_data in enumerate(self.valid_data_loader):

                batch_data["data"] = batch_data["data"].to(self.device)
                batch_data["target"] = batch_data["target"].to(self.device)
                batch_data["pad_mask"] = batch_data["pad_mask"].to(self.device)
                batch_data["bert_mask"] = batch_data["bert_mask"].to(self.device)
                    
                output = self.model(batch_data)
                pred = torch.argmax(output, dim=-1)
                loss = self.criterion(output, batch_data["target"], batch_data["bert_mask"])

                # compute term sentence level metrics
                term_predictions = get_term_predictions(pred, batch_data["target"], 
                                                        batch_data["bert_mask"], 
                                                        batch_data["sentences"], self.valid_data_loader.tags)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar("loss", loss.item())
                for met in self.sentence_metric_ftns:
                    self.writer.add_scalar(met.__name__, met(term_predictions["target"], 
                                                             term_predictions["prediction"]))
                    
                # update full epoch trackers
                epoch_target += term_predictions["target"]
                epoch_pred += term_predictions["prediction"]
                epoch_loss += loss.item()
                epoch_terms.update(term_predictions["predicted_terms"])
                
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        
        # compute epoch level sentence metrics
        log = {m.__name__: m(epoch_target, epoch_pred) for m in self.sentence_metric_ftns}
        
        # compute overall term identification metrics
        term_classifications = compute_term_categories(self.valid_data_loader.dataset.term_counts,
                                                       epoch_terms)
        log.update(**{m.__name__: m(term_classifications) for m in self.term_metric_ftns})
        
        log["loss"] = epoch_loss / batch_idx
        
        self.validation_classifications = term_classifications
        
        return log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
