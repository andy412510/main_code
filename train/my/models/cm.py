import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features  # prototype
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())  # Similarity between batch feature and prototype

        # debug for backward
        grad_inputs = outputs.mm(ctx.features)
        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)
        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:  # True
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


def anchor(batch_input, batch_labels, feature_memory, k, temp):
    instance_m = feature_memory.features.clone().detach()
    mat = torch.matmul(batch_input, instance_m.transpose(0, 1))
    positives = []
    negatives = []
    for i in range(batch_labels.size(0)):
        pos_labels = (feature_memory.labels == batch_labels[i])
        pos = mat[i, pos_labels]
        positives.append(pos[torch.argmax(pos)])
        neg_labels = torch.logical_and(feature_memory.labels != batch_labels[i], feature_memory.labels != -1)
        neg = torch.sort(mat[i, neg_labels], descending=False)[0]
        idx = neg[:k]
        negatives.append(idx)
    positives = torch.stack(positives)
    positives = positives.view(-1,1)
    negatives = torch.stack(negatives)
    anchor_out = torch.cat((positives, negatives), dim=1) / temp

    return anchor_out


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())

    def forward(self, inputs, targets, feature_memory, k):
        inputs = F.normalize(inputs, dim=1).cuda()  # batch data
        contrast_targets = torch.zeros([targets.size(0)]).cuda().long()
        # anchor_out = anchor(inputs, targets, feature_memory, k, self.temp)
        # anchor_loss = self.criterion(anchor_out, contrast_targets)

        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)

        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets)

        return loss


class HM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.criterion = nn.CrossEntropyLoss()
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())

    def forward(self, inputs, indexes):
        # inputs: B*2048, features: L*2048
        inputs = hm(inputs, indexes, self.features, self.momentum)
        inputs /= self.temp

        targets = self.labels[indexes].clone()
        labels = self.labels.clone()

        return F.nll_loss(torch.log(masked_sim+1e-6), targets)