import torch
import torch.nn.functional as F

def contrastive_clip_loss(z_graph, z_text, temperature=0.07):
    z_graph = F.normalize(z_graph, dim=-1)
    z_text  = F.normalize(z_text, dim=-1)

    logits = z_graph @ z_text.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_g2t = F.cross_entropy(logits, labels)
    loss_t2g = F.cross_entropy(logits.T, labels)

    return 0.5 * (loss_g2t + loss_t2g)
