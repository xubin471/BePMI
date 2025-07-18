import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(label, n_classes, requires_grad=True):
    """
    Args:
        label: (1, H, W)
        n_classes: 2
        requires_grad: True or False
    """
    device = label.device
    one_hot_label = torch.eye(
        n_classes, device=device, requires_grad=requires_grad)[label]  # (1, H, W, 2)
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)  # (1, 2, H, W) bg_mask and fg_mask

    return one_hot_label


class BoundaryLoss(nn.Module):

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            pred: the output from model (before softmax)
                shape (1, 2, H, W)
            gt: ground truth map
                shape (1, H, W)
        Return:
            boundary loss, averaged over mini-batch
        """

        n, c, _, _ = pred.shape

        # pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        one_hot_gt = one_hot(gt, c)  # (1, 2, H, W)

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt  # (1, 2, H, W)

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred  # (1, 2, H, W)

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)  # (1, 2, H, W)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)  # (1, 2, H, W)

        # reshape
        gt_b = gt_b.view(n, c, -1)  # (1, 2, HW)
        pred_b = pred_b.view(n, c, -1)  # (1, 2, HW)
        gt_b_ext = gt_b_ext.view(n, c, -1)  # (1, 2, HW)
        pred_b_ext = pred_b_ext.view(n, c, -1)  # (1, 2, HW)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss
