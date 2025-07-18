import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import SimpleITK as itk
import torch

class Visual:

    def __init__(self, model='Cow', organ_name="LIVER"):
        pass

    def update_status(self, model='BePMI', organ_name="LIVER"):
        organ = {
            "LIVER": {
                "name": "LIVER",
                "label_id": 1,

                "qry_patient_id": 38,
                "qry_slice": 5,

                "supp_patient_id": 37,
                "supp_slice": 3,
                "color": (0, 0, 0.7),
            },
            "RK": {
                "name": "RK",
                "label_id": 2,

                "qry_patient_id": 15,
                "qry_slice": 3,

                "supp_patient_id": 13,
                "supp_slice": 2,
                "color": (0, 0.7, 0),
            },
            "LK": {
                "name": "LK",
                "label_id": 3,

                "qry_patient_id": 36,
                "qry_slice": 6,

                "supp_patient_id": 33,
                "supp_slice": 7,
                "color": (0.7, 0, 0),
            },
            "SPLEEN": {
                "name": "SPLEEN",
                "label_id": 4,

                "qry_patient_id": 8,
                "qry_slice": 11,

                "supp_patient_id": 13,
                "supp_slice": 6,
                "color": (0, 1, 1),
            },

        }
        self.organ = organ[organ_name]
        self.abd_mri_pth = "./init"
        self.out_path = f"./sample/Abd_MRI/{model}/"
        self.model = model
        os.makedirs(self.out_path, exist_ok=True)


    def show1(self):
        qry_img_pth = os.path.join(self.abd_mri_pth, 'init', f"image_{self.organ['qry_patient_id']}.nii.gz")
        qry_gt_pth = os.path.join(self.abd_mri_pth, 'init', f"label_{self.organ['qry_patient_id']}.nii.gz")
        qry_pred_pth = os.path.join(self.abd_mri_pth, self.model,
                                    f"prediction_{self.organ['qry_patient_id']}_{self.organ['name']}.nii.gz")

        supp_img_pth = os.path.join(self.abd_mri_pth, 'init', f"image_{self.organ['supp_patient_id']}.nii.gz")
        supp_gt_pth = os.path.join(self.abd_mri_pth, 'init', f"label_{self.organ['supp_patient_id']}.nii.gz")

        # =============== get support init and qry init
        qry_img = itk.GetArrayFromImage(itk.ReadImage(qry_img_pth))  # (46, 256, 256)
        qry_gt = itk.GetArrayFromImage(itk.ReadImage(qry_gt_pth))  # (46, 256, 256)
        qry_pred = itk.GetArrayFromImage(itk.ReadImage(qry_pred_pth))

        supp_img = itk.GetArrayFromImage(itk.ReadImage(supp_img_pth))  # (46, 256, 256)
        supp_gt = itk.GetArrayFromImage(itk.ReadImage(supp_gt_pth))


        # ================== select valid slices
        qry_gt = 1 * (qry_gt == self.organ['label_id'])
        qry_idx = qry_gt.sum(axis=(1, 2)) > 0
        qry_img = qry_img[qry_idx]
        qry_gt = qry_gt[qry_idx]

        qry_img = qry_img[self.organ['qry_slice']] / 4.5
        qry_gt = qry_gt[self.organ['qry_slice']] * 200
        qry_pred = qry_pred[self.organ['qry_slice']] * 200

        supp_gt = 1 * (supp_gt == self.organ['label_id'])
        supp_idx = supp_gt.sum(axis=(1, 2)) > 0
        supp_img = supp_img[supp_idx]
        supp_gt = supp_gt[supp_idx]

        supp_img = supp_img[self.organ['supp_slice']] / 4.5
        supp_gt = supp_gt[self.organ['supp_slice']] * 200

        qry_img_gt = self.mask_add_image(qry_img, qry_gt, self.organ['color'])
        qry_img_pred = self.mask_add_image(qry_img, qry_pred, self.organ['color'])
        supp_img_gt = self.mask_add_image(supp_img, supp_gt, self.organ['color'])

        cv2.imwrite(
            self.out_path + f"/{self.organ['name']}_patient{self.organ['qry_patient_id']}_slice{self.organ['qry_slice']}_gt.png",
            qry_img_gt)
        cv2.imwrite(
            self.out_path + f"/{self.organ['name']}_patient{self.organ['qry_patient_id']}_slice{self.organ['qry_slice']}_pred.png",
            qry_img_pred)
        cv2.imwrite(
            self.out_path + f"/{self.organ['name']}_patient{self.organ['supp_patient_id']}_slice{self.organ['supp_slice']}_supp.png",
            supp_img_gt)
    def mask_add_image(self, image, mask, mask_color=(0, 0.8, 0)):
        """
        merge image with mask

        Args：
        - image: numpy, shape [256, 256]
        - mask: numpy, shape [256, 256]
        - mask_color: tuple, color (R, G, B)，range [0, 1]

        Return：
        - combined: Tensor, shape [batch, 3, 256, 256]
        """

        image = torch.tensor(image)
        image = torch.stack([image, image, image], dim=0).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)

        assert image.shape[2:] == mask.shape[1:], "Image and mask dimensions must match"
        assert image.shape[1] == 3, "Image must have 3 color channels"


        mask = mask.unsqueeze(1).float()  # [batch, 1, 256, 256]

        if image.min() != image.max():
            image = (image - image.min()) / (image.max() - image.min())  # 归一化到 [0, 1]
        else:
            image = image == 1

        if mask.min() != mask.max():
            mask = (mask - mask.min()) / (mask.max() - mask.min())  # 归一化到 [0, 1]
        else:
            mask = mask == 1


        mask_layer = torch.tensor(mask_color, device=image.device).view(1, 3, 1, 1)
        mask_layer = mask_layer * mask  # [batch, 3, 256, 256]

        # add mask to the image
        combined = image * (1 - 0.6 * mask) + mask_layer * 0.6 * mask
        combined *= 255
        return combined.squeeze(0).numpy().transpose(1, 2, 0)



visual = Visual()

methods = ["CoW","QNet","BePMI","GMRD"]

for method in methods:
    print(f"======== Method {method} begin ===============")
    visual.update_status(method, "LIVER")
    visual.show1()
    visual.update_status(method, "RK")
    visual.show1()
    visual.update_status(method, "LK")
    visual.show1()
    visual.update_status(method, "SPLEEN")
    visual.show1()
    print(f"======== Method { method } end ===============")

