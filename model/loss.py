import torch
import torch.nn as nn
import torch.nn.functional as F


class AngularContrastiveLoss(nn.Module):
    def __init__(
        self,
        margin,
        alpha=None,
        enableCL=True,
        temperature=0.06,  # temperature was not explored for this task
        device="cuda:0",
    ):
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.enableCL = enableCL
        self.margin = margin

        if self.enableCL:
            assert (
                alpha is not None
            ), f"You haven't provided alpha param.\nAlpha: {alpha}"
            self.alpha = alpha

    def forward(self, am_features, projection1=None, projection2=None, labels=None):
        if not self.enableCL:
            return self.amc(self.device, am_features, labels, self.margin)
        else:
            assert (projection1 is not None and projection2 is not None), \
                "You haven't provided feature projection for calculating the normal Contrastive Loss component."

            loss1 = self.scl(projection1, projection2, labels,
                             self.temperature, self.device)
            loss2 = self.amc(self.device, am_features, labels, self.margin)
            return loss1 * self.alpha + (1 - self.alpha) * loss2

    def scl(projection1, projection2, labels, temperature, device):
        projection1, projection2 = F.normalize(
            projection1), F.normalize(projection2)
        features = torch.cat(
            [projection1.unsqueeze(1), projection2.unsqueeze(1)], dim=1)
        batch_size = features.shape[0]

        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        else:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T), temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # for numerical stability

        mask = mask.repeat(contrast_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * contrast_count).view(-1, 1).to(device),
            0,
        )
        # or simply : logits_mask = torch.ones_like(mask) - torch.eye(50)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()

        return loss

    def amc(device, features, labels, margin):
        features = features.view(features.shape[0], -1)
        features = F.normalize(features, dim=1)
        labels = labels.contiguous().view(-1, 1)
        label_mask = torch.eq(labels, labels.T)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        diag_mask = torch.eye(label_mask.shape[0], dtype=torch.bool).to(device)
        label_mask = label_mask[~diag_mask].view(label_mask.shape[0], -1)
        similarity_matrix = similarity_matrix[~diag_mask].view(
            similarity_matrix.shape[0], -1
        )

        # select and combine multiple positives
        positives = similarity_matrix[label_mask.bool()]

        # select only the negatives the negatives
        negatives = similarity_matrix[~label_mask.bool()]

        # if S_ij = 0
        m = margin
        negatives = torch.clamp(negatives, min=-1 + 1e-7, max=1 - 1e-7)
        clip = torch.acos(negatives)
        l1 = torch.max(torch.zeros(clip.shape[0]).to(device), (m - clip))
        l1 = torch.sum(l1**2)

        # if S_ij = 1
        positives = torch.clamp(positives, min=-1 + 1e-7, max=1 - 1e-7)
        l2 = torch.acos(positives)
        l2 = torch.sum(l2**2)

        loss = (l1 + l2) / 50

        return loss
