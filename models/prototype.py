from torch import nn
import torch
import torch.nn.functional as F

class Prototype:
    @classmethod
    def compute_corralation(self,sup_fg_pts, qry_fg_pts, sup_bg_pts, qry_bg_pts):
        """
        Args:
            sup_fg_pts: (102, 512)
            qry_fg_pts: (102, 512)
            sup_bg_pts: (602, 512)
            qry_bg_pts: (602, 512)
        """
        d1 = torch.mean(sup_fg_pts, dim=0, keepdim=True)  # (1, 512)
        d2 = torch.mean(qry_fg_pts, dim=0, keepdim=True)  # (1, 512)
        b1 = torch.mean(sup_bg_pts, dim=0, keepdim=True)  # (1, 512)
        b2 = torch.mean(qry_bg_pts, dim=0, keepdim=True)  # (1, 512)

        d1 = F.normalize(d1, dim=-1)  # (1, 512)
        d2 = F.normalize(d2, dim=-1)  # (1, 512)
        b1 = F.normalize(b1, dim=-1)  # (1, 512)
        b2 = F.normalize(b2, dim=-1)  # (1, 512)

        fg_intra = torch.matmul(d1, d2.transpose(0, 1)).squeeze(0).squeeze(0)
        bg_intra = torch.matmul(b1, b2.transpose(0, 1)).squeeze(0).squeeze(0)
        intra_loss = 2 - fg_intra - bg_intra

        zero = torch.zeros(1).squeeze(0)
        sup_inter = torch.matmul(d1, b1.transpose(0, 1))  # (1, 1)
        qry_inter = torch.matmul(d2, b2.transpose(0, 1))  # (1, 1)
        inter_loss = torch.max(zero, torch.mean(sup_inter)) + torch.max(zero, torch.mean(qry_inter))

        # return torch.sigmoid(intra_loss + inter_loss) * 20
        return intra_loss + inter_loss