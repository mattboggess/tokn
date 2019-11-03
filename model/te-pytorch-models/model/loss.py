import torch.nn.functional as F
import torch

def nll_loss(output, target, bert_mask):
    """ 
    Cross entropy loss with extra bert tokens masked out.
    
    Parameters
    ----------
    output: torch.Tensor
      batch_size x max_sentence_length x num_classes model output as log probabilities
    target: torch.Tensor
      batch_size x max_sentence_length target labels in [0, num_classes - 1] 
    bert_mask: torch.Tensor
      batch_size x max_sentence_length mask that is 1 for original tokens and 0 for added tokens 
    """
    output = output.view(output.shape[0] * output.shape[1], -1) 
    target = target.view(target.shape[0] * target.shape[1]) 
    bert_mask = bert_mask.view(bert_mask.shape[0] * bert_mask.shape[1])
    loss = F.nll_loss(output, target, reduction="none")
    loss = torch.sum(loss * bert_mask) / torch.sum(bert_mask)
    return loss
