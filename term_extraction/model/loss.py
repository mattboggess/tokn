import torch.nn.functional as F
import torch

def nll_loss(output, target, bert_mask, class_weights, model=None):
    """
    Cross entropy loss with extra bert wordpiece tokens and padding tokens masked out.

    Parameters
    ----------
    output: torch.Tensor
      batch_size x max_sentence_length x num_classes model output as log probabilities
    target: torch.Tensor
      batch_size x max_sentence_length target labels in [0, num_classes - 1]
    bert_mask: torch.Tensor
      batch_size x max_sentence_length mask that is 1 for original tokens and 0 for 
      added wordpiece or padding tokens (walking -> walk #ing <pad> = [1, 0, 0])
    class_weights: torch.Tensor
      num_tags x 1 weights used to adjust for class imbalance
    
    Returns
    -------
    int
        mean negative log likelihood cross entropy loss for batch ignoring masked tokens
    """
    bert_mask = bert_mask.to(torch.float32)
    output = output.view(output.shape[0] * output.shape[1], -1)
    target = target.view(target.shape[0] * target.shape[1])
    bert_mask = bert_mask.view(bert_mask.shape[0] * bert_mask.shape[1])
    loss = F.nll_loss(output, target, reduction='none', weight=class_weights)
    loss = torch.sum(loss * bert_mask) / torch.sum(bert_mask)
    return loss

def crf_loss(emissions, target, bert_mask, class_weights, model):
    """
    Conditional random field negative log likelihood with extra Bert tokens masked out.
    """
    #ll = 0
    #for i in range(emissions.shape[0]):
    #    mask = bert_mask[i, :] == 1
    #    ems = emissions[i, mask, :].unsqueeze(0)
    #    tags = target[i, mask].unsqueeze(0)
    #    ll += model.crf(ems, tags)
        
    return -model.crf(emissions[:, 1:, :], target[:, 1:], mask=bert_mask[:, 1:].to(torch.uint8))
