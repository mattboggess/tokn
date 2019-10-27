import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from torchvision.utils import make_grid
from sklearn.metrics import classification_report


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
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
        self.train_metrics.reset()
        
        epoch_target = []
        epoch_pred = []
        epoch_word_pairs = []
        epoch_loss = 0
        
        for batch_idx, (data, target, word_pair) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            print(target)

            self.optimizer.zero_grad()
            output = self.model(data)
            with torch.no_grad():
                pred = torch.argmax(output, dim=1)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            
            # add graph
            #if batch_idx == 0:
            #    self.writer.add_graph(self.model, data)
            
            # accumulate epoch quantities 
            epoch_target += [t.item() for t in target]
            epoch_pred += [p.item() for p in pred] 
            epoch_word_pairs += word_pair
            epoch_loss += loss.item()
            
            # update metrics
            self.writer.add_scalar("loss", epoch_loss / (batch_idx + 1))
            for met in self.metric_ftns:
                self.writer.add_scalar(met.__name__, met(epoch_target, epoch_pred))
            
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = {m.__name__: m(epoch_target, epoch_pred) for m in self.metric_ftns}

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            
            epoch_target = []
            epoch_pred = []
            epoch_word_pairs = []
            epoch_loss = 0
            
            for batch_idx, (data, target, word_pair) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                pred = torch.argmax(output, dim=1)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                
                # accumulate epoch quantities 
                epoch_target += [t.item() for t in target]
                epoch_pred += [p.item() for p in pred] 
                epoch_word_pairs += word_pair
                epoch_loss += loss.item()
                
                # update metrics
                self.writer.add_scalar("loss", epoch_loss / (batch_idx + 1))
                for met in self.metric_ftns:
                    self.writer.add_scalar(met.__name__, met(epoch_target, epoch_pred))
                    
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        log = {m.__name__: m(epoch_target, epoch_pred) for m in self.metric_ftns}
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
