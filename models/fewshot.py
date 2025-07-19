
# from visualize import visualize_feature_space
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Res101Encoder
from models.decoder import MLP, Decoder
import numpy as np
import random
import cv2
from boundary_loss import BoundaryLoss
from .prototype import Prototype
from .attention import Channel_att as Channel_att
from .util import kmeans
import matplotlib.pyplot as plt
class FewShotSeg(nn.Module):
    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()
        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.criterion = nn.NLLLoss()
        self.criterion_b = BoundaryLoss(theta0=3, theta=5)
        self.criterion_MSE = nn.MSELoss()
        self.fg_num = 70
        self.fg_edge_num = 30
        self.bg_num = 540
        self.bg_edge_num =  60
        self.mlp_fg = MLP(256, self.fg_num)
        self.mlp_bg = MLP(256, self.bg_num)


        self.decoder1 = Decoder(self.fg_num+self.fg_edge_num)
        self.decoder2 = Decoder(self.bg_num+self.bg_edge_num)
        # self.supp_decoder = Supp_Decoder()

        # self.fg_enhance = CrossProtoAttention()
        # self.bg_enhance = CrossProtoAttention()
        self.channel_attn = Channel_att()
        self.testing = False



    def forward(self, supp_imgs, supp_mask, qry_imgs, train=False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W]
            supp_mask: foreground masks for support images
                way x shot x [B x H x W]
            qry_imgs: query images
                N x [B x 3 x H x W]
            train: whether to train model or not
        """
        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # (B, way, shot, H, W)

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)  # (2, 3, 256, 256)
        img_fts, tao = self.encoder(imgs_concat)

        # supp_fts: list[tensor(supp_bs, way, shot, 512, 64, 64) + tensor(supp_bs, way, shot, 512, 32, 32)]
        supp_fts = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view(
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        supp_fts = supp_fts[0]  # (supp_bs, way, shot, 512, 64, 64)
        # qry_fts: list[tensor(qry_bs, N, 512, 64, 64) + tensor(qry_bs, N, 512, 32, 32)]
        qry_fts = [img_fts[dic][self.n_ways * self.n_shots * supp_bs:].view(
            qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        qry_fts = qry_fts[0]  # (qry_bs, N, 512, 64, 64)

        ###### Get threshold ######
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]  # t for query features (1, 1)
        self.thresh_pred = [self.t for _ in range(self.n_ways)]  # way x [1 x 1]

        ###### Compute loss ######
        align_loss = torch.zeros(1).to(self.device)
        b_loss = torch.zeros(1).to(self.device)
        ssp_loss = torch.zeros(1).to(self.device)
        aux_loss = torch.zeros(1).to(self.device)
        outputs = []
        for epi in range(supp_bs):
            ###### Extract prototypes ######
            fg_pts = [[self.get_fg_pts(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot], None)
                       for shot in range(self.n_shots)] for way in range(self.n_ways)]  # way x shot x [152, 512]
            fg_pts = self.get_all_prototypes(fg_pts)  # way x [152, 512]

            bg_pts = [[self.get_bg_pts(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot], None)
                       for shot in range(self.n_shots)] for way in range(self.n_ways)]  # way x shot x [702, 512]
            bg_pts = self.get_all_prototypes(bg_pts)  # way x [702, 512]

            ###### Get query predictions ######
            fg_sim = torch.stack(
                [self.get_fg_sim(qry_fts[epi], fg_pts[way]) for way in range(self.n_ways)], dim=1).squeeze(0)
            bg_sim = torch.stack(
                [self.get_bg_sim(qry_fts[epi], bg_pts[way]) for way in range(self.n_ways)], dim=1).squeeze(0)

            fg_pred = F.interpolate(fg_sim, size=img_size, mode='bilinear', align_corners=True)
            bg_pred = F.interpolate(bg_sim, size=img_size, mode='bilinear', align_corners=True)
            preds = torch.cat([bg_pred, fg_pred], dim=1)  # (1, 2, 256, 256)
            preds = torch.softmax(preds, dim=1)  # (1, 2, 256, 256)

            outputs.append(preds)  # supp_bs x [1 x 2 x 256 x 256]
            if train:
                align_loss_epi, aux_loss_epi, b_loss_epi, ssp_loss_epi = self.align_aux_Loss(supp_fts[epi], qry_fts[epi], supp_mask[epi], preds,
                                                                                             fg_pts, bg_pts, None)
                align_loss += align_loss_epi
                aux_loss += aux_loss_epi
                b_loss += b_loss_epi
                ssp_loss += ssp_loss_epi


        output = torch.stack(outputs, dim=1)  # N x B x (1 + way) x H x W
        output = output.view(-1, *output.shape[2:])  # (1, 2, 256, 256)

        if train:
            return output, align_loss / supp_bs, aux_loss / supp_bs, b_loss / supp_bs, ssp_loss / supp_bs
        else:
            return output

    def align_aux_Loss(self, supp_fts, qry_fts, fore_mask, pred, sup_fg_pts, sup_bg_pts, self_pred):
        """
        Args:
            supp_fts: (way, shot, 512, 64, 64)
            qry_fts: (N, 512, 64, 64)
            fore_mask: (way, shot, 256, 256)
            pred: (1, 2, 256, 256)
            sup_fg_pts: way x [102 x 512]
            sup_bg_pts: way x [602 x 512]
            self_pred: (1, 2, 256, 256)
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # (1, 256, 256)
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]  # 2 x [1 x 256 x 256] (True-False)
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (2, 1, 256, 256) (0-1)

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        loss_aux = torch.zeros(1).to(self.device)
        b_loss = torch.zeros(1).to(self.device)
        ssp_loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                fg_pts_ = [[self.get_fg_pts(qry_fts, pred_mask[way + 1], None)]]  # 1 x 1 x [102 x 512]
                fg_pts_ = self.get_all_prototypes(fg_pts_)  # 1 x [102 x 512]
                bg_pts_ = [[self.get_bg_pts(qry_fts, pred_mask[way + 1], None)]]  # 1 x 1 x [102 x 512]
                bg_pts_ = self.get_all_prototypes(bg_pts_)  # 1 x [102 x 512]

                loss_aux += self.get_aux_loss(sup_fg_pts[way], fg_pts_[way], sup_bg_pts[way], bg_pts_[way])

                # Get predictions
                supp_pred = self.get_fg_sim(supp_fts[way, [shot]], fg_pts_[way])  # (1, 1, 64, 64)
                bg_pred_ = self.get_bg_sim(supp_fts[way, [shot]], bg_pts_[way])  # (1, 1, 64, 64)
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                bg_pred_ = F.interpolate(bg_pred_, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)

                # Combine predictions
                preds = torch.cat([bg_pred_, supp_pred], dim=1)  # (1, 2, 256, 256)
                preds = torch.softmax(preds, dim=1)  # (1, 2, 256, 256)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(preds, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

                # b_loss += self.criterion_b(torch.clamp(preds, eps, 1 - eps),
                #                            supp_label[None, ...].long()) / n_shots / n_ways
                # ssp_log_prob = torch.log(torch.clamp(self_pred, eps, 1 - eps))
                # ssp_loss += self.criterion(ssp_log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss, loss_aux, b_loss, ssp_loss

    def get_fg_pts(self, features, mask, pred_mask):
        """
        Args:
        features: (1, 512, 64, 64)
        mask: (1, 256, 256)
        pred_mask: (1, 256, 256)
        """
        features_trans = F.interpolate(features, size=mask.shape[-2:], mode='bilinear', align_corners=True)  # (1, 512, 256, 256)

        ie_mask = mask.squeeze(0) - torch.tensor(cv2.erode(mask.squeeze(0).cpu().numpy(), np.ones((3, 3), dtype=np.uint8), iterations=2)).to(self.device)
        ie_mask = ie_mask.unsqueeze(0)  # (1, 256, 256)
        ie_prototype = torch.sum(features_trans * ie_mask[None, ...], dim=(-2, -1)) \
                       / (ie_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # (1, 512)
        origin_prototype = torch.sum(features_trans * mask[None, ...], dim=(-2, -1)) \
                           / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # (1, 512)

        fg_fts = self.get_fg_fts(features_trans, mask)  # (1, 512, 256, 256)
        fg_prototypes = self.mlp_fg(fg_fts.view(512, 256 * 256)).permute(1, 0)  # (100, 512)

        if ie_mask.sum()>0:
            if ie_mask.sum() < 100:
                ie_fg = self.get_random_pts(features_trans, ie_mask, self.fg_edge_num)  # (30, 512)
            else:
                boundary_coords = torch.where(ie_mask.squeeze() == 1)
                boundary_features = features_trans[0, :, boundary_coords[0], boundary_coords[1]].T  # [M,512]
                with torch.no_grad():
                    is_ok, _, _, ie_fg = kmeans(boundary_features.unsqueeze(0), self.fg_edge_num)
                if is_ok:
                    ie_fg = ie_fg.squeeze(0)
                else:
                    k = random.sample(range(len(fg_prototypes)), self.fg_edge_num)
                    ie_fg = fg_prototypes[k]
        else:
            k = random.sample(range(len(fg_prototypes)), self.fg_edge_num)
            ie_fg = fg_prototypes[k]
        """
        
        """
        # visualize_feature_space(boundary_features,prototypes=ie_fg)

        fg_prototypes = torch.cat([fg_prototypes,ie_fg],dim=0)

        fg_prototypes = torch.cat([fg_prototypes, ie_prototype,origin_prototype], dim=0)  # (152, 512)
        if self.testing:
            fg_prototypes = F.normalize(fg_prototypes,dim=-1)
        return fg_prototypes ## (152, 512)

    def get_bg_pts(self, features, mask, pred_mask):
        """
        Args:
            features: (1, 512, 64, 64)
            mask: (1, 256, 256)
            pred_mask: (1, 256, 256)
        """
        bg_mask = 1 - mask  # (1, 256, 256)
        features_trans = F.interpolate(features, size=bg_mask.shape[-2:], mode='bilinear', align_corners=True)  # (1, 512, 256, 256)

        oe_mask = torch.tensor(cv2.dilate(mask.squeeze(0).cpu().numpy(), np.ones((3, 3), dtype=np.uint8), iterations=2)).to(self.device) - mask.squeeze(0)
        oe_mask = oe_mask.unsqueeze(0)  # (1, 256, 256)
        oe_prototype = torch.sum(features_trans * oe_mask[None, ...], dim=(-2, -1)) \
                       / (oe_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # (1, 512)
        origin_prototype = torch.sum(features_trans * bg_mask[None, ...], dim=(-2, -1)) \
                           / (bg_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # (1, 512)

        bg_fts = self.get_fg_fts(features_trans, bg_mask)  # (1, 512, 256, 256)
        bg_prototypes = self.mlp_bg(bg_fts.view(512, 256 * 256)).permute(1, 0)  # (600, 512)

        if oe_mask.sum()>0:
            if oe_mask.sum() < 200:
                oe_bg = self.get_random_pts(features_trans, oe_mask, self.bg_edge_num)  # (30, 512)
            else:
                boundary_coords = torch.where(oe_mask.squeeze() == 1)
                boundary_features = features_trans[0, :, boundary_coords[0], boundary_coords[1]].T  # [M,512]
                with torch.no_grad():
                    is_ok,_, _, oe_bg = kmeans(boundary_features.unsqueeze(0), self.bg_edge_num)
                if is_ok:
                    oe_bg = oe_bg.squeeze(0)
                else:
                    k = random.sample(range(len(bg_prototypes)), self.bg_edge_num)
                    oe_bg = bg_prototypes[k]

        else:
            k = random.sample(range(len(bg_prototypes)), self.bg_edge_num)
            oe_bg = bg_prototypes[k]

        # visualize_feature_space(boundary_features,prototypes=oe_bg)

        bg_prototypes = torch.cat([bg_prototypes,oe_bg],dim=0)
        bg_prototypes = torch.cat([bg_prototypes,oe_prototype,origin_prototype], dim=0)  # (602, 512)
        
        if self.testing:
            bg_prototypes = F.normalize(bg_prototypes,dim=-1)
        return bg_prototypes # (602, 512)

    def get_random_pts(self, features_trans, mask, n_prototype):
        """
        Args:
            features_trans: (1, 512, 256, 256)
            mask: (1, 256, 256)
            n_prototype: int
        """
        features_trans = features_trans.squeeze(0)  # (512, 256, 256)
        features_trans = features_trans.permute(1, 2, 0)  # (256, 256, 512)
        features_trans = features_trans.view(features_trans.shape[-2] * features_trans.shape[-3],
                                             features_trans.shape[-1])  # (256 * 256, 512)
        mask = mask.squeeze(0).view(-1)  # (256 * 256)
        features_trans = features_trans[mask == 1]  # (n_fg, 512)
        if len(features_trans) >= n_prototype:
            k = random.sample(range(len(features_trans)), n_prototype)
            prototypes = features_trans[k]  # (n_prototype, 512)
        else:
            if len(features_trans) == 0:
                prototypes = torch.zeros(n_prototype, 512).to(self.device)  # (n_prototype, 512)
            else:
                r = n_prototype // len(features_trans)
                k = random.sample(range(len(features_trans)), (n_prototype - len(features_trans)) % len(features_trans))
                prototypes = torch.cat([features_trans for _ in range(r)], dim=0)
                prototypes = torch.cat([features_trans[k], prototypes], dim=0)  # (n_prototype, 512)
                

        return prototypes

    def get_fg_fts(self, fts, mask):
        """
        Args:
            fts: (1, 512, 256, 256)
            mask: (1, 256, 256)
        """
        _, c, h, w = fts.shape
        # select masked fg features
        fg_fts = fts * mask[None, ...]  # (1, 512, 256, 256)
        bg_fts = torch.ones_like(fts) * mask[None, ...]  # (1, 512, 256, 256)
        mask_ = mask.view(-1)
        n_pts = len(mask_) - len(mask_[mask_ == 1])
        select_pts = self.get_random_pts(fts, mask, n_pts)  # (n_pts, 512)
        index = bg_fts == 0
        fg_fts[index] = select_pts.permute(1, 0).reshape(512*n_pts)  # (1, 512, 256, 256)

        return fg_fts

    def get_all_prototypes(self, fg_fts):
        """
        Args:
            fg_fts: way x shot x [all x 512]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        prototypes = [sum([shot for shot in way]) / n_shots for way in fg_fts]  # way x [all x 512]

        return prototypes

    def get_fg_sim(self, fts, prototypes):
        """
        Args:
            fts: (1, 512, 64, 64)
            prototypes: (102, 512)
        """
        # 基于原型的通道注意力
        fts = self.channel_attn(fts, prototypes[-1:])

        # # 基于原型的注意力
        # fts = self.fg_enhance(fts,prototypes)

        feat_norm = F.normalize(fts, p=2, dim=1)
        protos_norm = F.normalize(prototypes, p=2, dim=1)[:, :, None, None]
        fg_sim = F.conv2d(feat_norm, protos_norm)  # 组卷积实现相似度
        fg_sim = self.decoder1(fg_sim)  # (1, 1, 64, 64)

        return fg_sim

    def get_bg_sim(self, fts, prototypes):
        """
        Args:
            fts: (1, 512, 64, 64)
            prototypes: (602, 512)
        """
        # 基于原型的通道注意力
        fts = self.channel_attn(fts, prototypes[-1:])

        # 基于原型的注意力
        # fts = self.bg_enhance(fts, prototypes)

        feat_norm = F.normalize(fts, p=2, dim=1)
        protos_norm = F.normalize(prototypes, p=2, dim=1)[:, :, None, None]
        bg_sim = F.conv2d(feat_norm, protos_norm)  # 组卷积实现相似度
        bg_sim = self.decoder2(bg_sim)  # (1, 1, 64, 64)

        return bg_sim

    def get_aux_loss(self, sup_fg_pts, qry_fg_pts, sup_bg_pts, qry_bg_pts):
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

        return intra_loss + inter_loss

    def test(self, supp_img, supp_mask, qry_imgs, query_label=None,train=False):
        """
        Args:
            qry_imgs [num x 1 x 3 x 256 x 256]
            supp_img:
                tensor [1 3 256 256]
            supp_mask:
                tensor [1 256 256]
        """
        self.testing = True
        img_size = supp_mask.shape[-2:]
        img_concat = torch.cat([supp_img, qry_imgs.squeeze(1)], dim=0)
        fts, tao = self.encoder(img_concat)
        qry_fts = fts['down2'][len(supp_img):].unsqueeze(1) # (num 1 512 h w)
        supp_ft = fts['down2'][:len(supp_img)] # (1 512 h w)

        fg_pts = self.get_fg_pts(supp_ft,supp_mask,None) #(102 512)
        bg_pts = self.get_bg_pts(supp_ft,supp_mask,None) #(602 512)

        preds = []
        for i_ in range(len(qry_fts)):
            fg_sim = self.get_fg_sim(qry_fts[i_], fg_pts)
            bg_sim = self.get_bg_sim(qry_fts[i_], bg_pts)
            fg_pred = F.interpolate(fg_sim, size=img_size, mode='bilinear', align_corners=True)
            bg_pred = F.interpolate(bg_sim, size=img_size, mode='bilinear', align_corners=True)
            pred = torch.cat([bg_pred, fg_pred], dim=1)
            pred = torch.argmax(torch.softmax(pred, dim=1),dim=1)
            preds.append(pred.float()) #(1 2 256 256)


        for _ in range(1):
            qry_fg_protos = [self.get_fg_pts(qry_fts[i_], preds[i_].float(), preds[i_].float()) for i_ in
                             range(len(qry_fts))]
            qry_bg_protos = [self.get_bg_pts(qry_fts[i_], preds[i_].float(), preds[i_].float()) for i_ in
                             range(len(qry_fts))]
            protos_corr = [Prototype.compute_corralation(fg_pts, qry_fg_protos[i_], bg_pts, qry_bg_protos[i_]) for i_ in
                           range(len(qry_fts))]
            protos_corr = torch.tensor(protos_corr).cuda()
            # protos_corr = torch.softmax(protos_corr, dim=0)
            protos_corr = torch.sigmoid(protos_corr)-0.3

            score_max_index = torch.argmin(protos_corr, dim=0)

            left_list = sorted([i for i in range(0, score_max_index)], reverse=True)
            right_list = [i for i in range(score_max_index + 1, len(qry_imgs))]

            for i in left_list:
                rate = float(1 - protos_corr[i])
                fg_pts_ = self.get_fg_pts(qry_fts[i + 1], preds[i + 1], None)  # (102 512)
                fg_pts_ = self.get_protos_test(fg_pts, fg_pts_, rate)

                bg_pts_ = self.get_bg_pts(qry_fts[i + 1], preds[i + 1], None)  # (602 512)
                bg_pts_ = self.get_protos_test(bg_pts, bg_pts_, rate)

                fg_sim_ = self.get_fg_sim(qry_fts[i], fg_pts_)
                bg_sim_ = self.get_bg_sim(qry_fts[i], bg_pts_)
                fg_pred_ = F.interpolate(fg_sim_, size=img_size, mode='bilinear', align_corners=True)
                bg_pred_ = F.interpolate(bg_sim_, size=img_size, mode='bilinear', align_corners=True)
                pred_ = torch.cat([bg_pred_, fg_pred_], dim=1)
                pred_ = torch.argmax(torch.softmax(pred_, dim=1), dim=1)
                preds[i] = pred_.float()

            for i in right_list:
                rate = float(1 - protos_corr[i])
                fg_pts_ = self.get_fg_pts(qry_fts[i - 1], preds[i - 1], None)  # (102 512)
                fg_pts_ = self.get_protos_test(fg_pts, fg_pts_, rate)

                bg_pts_ = self.get_bg_pts(qry_fts[i - 1], preds[i - 1], None)  # (602 512)
                bg_pts_ = self.get_protos_test(bg_pts, bg_pts_, rate)

                fg_sim_ = self.get_fg_sim(qry_fts[i], fg_pts_)
                bg_sim_ = self.get_bg_sim(qry_fts[i], bg_pts_)
                fg_pred_ = F.interpolate(fg_sim_, size=img_size, mode='bilinear', align_corners=True)
                bg_pred_ = F.interpolate(bg_sim_, size=img_size, mode='bilinear', align_corners=True)
                pred_ = torch.cat([bg_pred_, fg_pred_], dim=1)
                pred_ = torch.argmax(torch.softmax(pred_, dim=1), dim=1)
                preds[i] = pred_.float()

        return torch.cat(preds, dim=0)

    def get_protos_test(self, sup_protos, protos, sup_rate):
        """
            Args:
                sup_protos: (n1+2 512)
                protos: (n1+2 512)
                rate: [0.0-1.0]
        """
        length = sup_protos.shape[0] - 2
        return torch.cat([protos[:int(length * sup_rate)], sup_protos[int(length * sup_rate):]], dim=0)
