import torch
import torch.nn.functional as F


def dis_criterion(fake_op, real_op):
    """
    Hinge loss function for discriminator
    """
    return torch.mean(F.relu(1.0 - real_op)) + torch.mean(F.relu(1.0 + fake_op))


def gen_criterion(dis_preds, ctc_loss):
    """
    Hinge loss function for generator
    """
    return ctc_loss - torch.mean(dis_preds)
    # return -torch.mean(dis_preds)


def compute_ctc_loss(criterion, ip, tgt, tgt_lens):
    """
    CTC loss function for the OCR network
    """
    ip_lens = torch.full(size=(ip.shape[1],), fill_value=ip.shape[0])
    return criterion(ip, tgt, ip_lens, tgt_lens)
