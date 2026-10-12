import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights


def stable_normalize(values, dim=-1, eps=1e-12):
    scale = values.abs().amax(dim=dim, keepdim=True).clamp_min(eps)
    scaled = values / scale
    length = scaled.square().sum(dim=dim, keepdim=True).clamp_min(1e-24).sqrt()
    return scaled / length.clamp_min(eps / scale)


def normalize_cam(cam):
    cam = cam.clamp_min(0)
    cam = cam - cam.amin(dim=(-2, -1), keepdim=True)
    return cam / cam.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)


def cluster_features(values, count, iterations=10, metric='cosine'):
    if values.shape[0] == 0:
        raise ValueError('No activated features available')
    with torch.no_grad():
        unit = values.detach().float() if metric == 'euclidean' else stable_normalize(values.detach().float(), dim=-1)
        selected = [0]
        distance = (unit - unit[0]).square().sum(-1) if metric == 'euclidean' else 1 - unit @ unit[0]
        for _ in range(1, min(count, len(unit))):
            index = int(distance.argmax())
            selected.append(index)
            new_distance = (unit - unit[index]).square().sum(-1) if metric == 'euclidean' else 1 - unit @ unit[index]
            distance = torch.minimum(distance, new_distance)
        centers = unit[selected]
        for _ in range(iterations):
            assignment = torch.cdist(unit, centers).argmin(-1) if metric == 'euclidean' else (unit @ centers.T).argmax(-1)
            updated = []
            for k in range(len(centers)):
                members = unit[assignment == k]
                updated.append(members.mean(0) if len(members) else centers[k])
            centers = torch.stack(updated) if metric == 'euclidean' else stable_normalize(torch.stack(updated), dim=-1)
        assignment = torch.cdist(unit, centers).argmin(-1) if metric == 'euclidean' else (unit @ centers.T).argmax(-1)
    result = []
    for k in range(len(centers)):
        members = values[assignment == k]
        result.append(members.mean(0) if len(members) else values[selected[k]])
    return stable_normalize(torch.stack(result), dim=-1)


class ROPMB(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.classes = cfg['classes']
        self.kg = cfg['global_prototypes']
        self.ki = cfg['local_prototypes']
        backbone = resnet50(weights=None)
        if pretrained:
            backbone.load_state_dict(torch.load(cfg['backbone_weights'], map_location='cpu'))
        backbone.layer4[0].conv2.stride = (1, 1)
        backbone.layer4[0].downsample[0].stride = (1, 1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.classifier = nn.Conv2d(2048, self.classes, 1, bias=False)
        self.project = nn.Conv2d(2048, cfg['embedding_dim'], 1, bias=False)
        self.memory = nn.Parameter(stable_normalize(torch.randn(self.classes, self.kg, cfg['embedding_dim']), dim=-1))
        self.gcn = nn.Linear(cfg['embedding_dim'], cfg['embedding_dim'], bias=False)
        nn.init.eye_(self.gcn.weight)
        self.register_buffer('prior', torch.eye(self.classes))
        self.register_buffer('initialized', torch.zeros(self.classes, dtype=torch.bool))

    def encode(self, images):
        feature = self.encoder(images)
        if not torch.isfinite(feature).all() or feature.detach().abs().max() > 1e8:
            raise FloatingPointError("Backbone feature scale is invalid")
        projected = self.project(feature)
        if self.cfg.get('normalize_pixel_features', True):
            projected = stable_normalize(projected, dim=1)
        return projected, normalize_cam(self.classifier(feature)), feature

    def graph(self):
        bank = stable_normalize(self.memory.flatten(0, 1), dim=-1)
        prior = self.prior.repeat_interleave(self.kg, 0).repeat_interleave(self.kg, 1)
        score = (bank @ bank.T).relu() * prior
        score = score.masked_fill(torch.eye(len(bank), device=bank.device, dtype=torch.bool), 0)
        indices = score.topk(min(self.cfg['graph_neighbors'], len(bank) - 1), dim=-1).indices
        adjacency = torch.zeros_like(score).scatter(1, indices, score.gather(1, indices))
        adjacency = adjacency + torch.eye(len(bank), device=bank.device)
        inv_degree = adjacency.sum(1).clamp_min(1e-6).rsqrt()
        normalized = inv_degree[:, None] * adjacency * inv_degree[None, :]
        if self.cfg.get('use_graph', True):
            optimized = self.gcn(normalized @ bank)
        else:
            optimized = bank
        return optimized.view_as(self.memory), adjacency

    def extract(self, feature, seed, labels):
        seed = F.interpolate(seed, size=feature.shape[-2:], mode='bilinear', align_corners=False)
        records = []
        for b in range(len(feature)):
            flat = feature[b].flatten(1).T
            for c in labels[b].nonzero().flatten().tolist():
                active = seed[b, c].flatten() > self.cfg['activation_threshold']
                if not active.any():
                    continue
                local = cluster_features(flat[active], self.ki, self.cfg['kmeans_iterations'], self.cfg.get('clustering_metric', 'cosine'))
                local = stable_normalize(F.linear(local, self.project.weight[:, :, 0, 0]), dim=-1)
                records.append((b, c, local))
        return records

    def forward(self, images, seeds, labels):
        feature, classifier_cam, raw_feature = self.encode(images)
        optimized, adjacency = self.graph()
        baseline = optimized.mean(1) * self.ki
        if self.cfg.get('absent_class_response', 'global') == 'zero':
            cams = feature.new_zeros((len(images), self.classes, *feature.shape[-2:]))
        else:
            cams = torch.einsum('bdhw,cd->bchw', feature, baseline).relu()
        records = self.extract(raw_feature, seeds, labels)
        consistency = feature.sum() * 0
        updates = []
        for b, c, local in records:
            similarities = stable_normalize(local, dim=-1) @ stable_normalize(optimized[c], dim=-1).T
            refined = (similarities / self.cfg['temperature']).softmax(-1) @ optimized[c]
            activation = torch.einsum('dhw,kd->khw', feature[b], refined).relu().sum(0)
            cams = cams.clone()
            cams[b, c] = activation
            consistency = consistency + (local - refined).square().sum()
            for slot, value in zip(similarities.detach().argmax(-1).tolist(), local.detach()):
                updates.append((c, slot, value.cpu()))
        bank = stable_normalize(self.memory, dim=-1)
        similarities = bank @ bank.transpose(-1, -2)
        mask = ~torch.eye(self.kg, device=bank.device, dtype=torch.bool)
        diversity = (similarities - self.cfg['diversity_threshold']).relu()[:, mask].sum()
        nodes = optimized.flatten(0, 1)
        distances = (nodes[:, None] - nodes[None, :]).square().sum(-1)
        smoothness = (adjacency * distances).sum()
        logits = cams.mean((-2, -1))
        classification = F.multilabel_soft_margin_loss(logits, labels)
        losses = {'classification': classification, 'consistency': consistency / len(images), 'diversity': diversity, 'smoothness': smoothness}
        normalized_cams = normalize_cam(cams)
        if self.cfg.get('auxiliary_classification_weight', 0):
            classifier_logits = self.classifier(raw_feature).mean((-2, -1))
            losses['auxiliary_classification'] = F.multilabel_soft_margin_loss(classifier_logits, labels)
        if self.cfg.get('cam_alignment_weight', 0):
            target = F.interpolate(seeds, size=cams.shape[-2:], mode='bilinear', align_corners=False)
            target = target / target.amax((-2, -1), keepdim=True).clamp_min(1e-6)
            valid = F.interpolate((images.detach().abs().sum(1, keepdim=True) > 0).float(), size=cams.shape[-2:], mode='nearest')
            supervision = valid * labels[:, :, None, None]
            losses['cam_alignment'] = ((normalized_cams - target).square() * supervision).sum() / supervision.sum().clamp_min(1)
        return {'cams': normalized_cams, 'classifier_cams': classifier_cam, 'losses': losses, 'updates': updates, 'records': len(records), 'health': {'raw_feature_max': raw_feature.detach().abs().max(), 'projected_norm_mean': feature.detach().square().sum(1).sqrt().mean(), 'projected_norm_max': feature.detach().square().sum(1).sqrt().amax(), 'cam_max': cams.detach().amax(), 'active_cam_fraction': (cams.detach().amax((-2, -1)) > 1e-6).float().mean()}}

    @torch.no_grad()
    def update_memory(self, updates):
        for c, slot, value in updates:
            self.memory[c, slot].mul_(self.cfg['momentum']).add_(value.to(self.memory), alpha=1 - self.cfg['momentum'])
        self.memory.copy_(stable_normalize(self.memory, dim=-1))


def project_conflicting(diversity, smoothness):
    dot = sum((a * b).sum() for a, b in zip(diversity, smoothness))
    norm = sum(b.square().sum() for b in smoothness).clamp_min(1e-12)
    coefficient = torch.minimum(dot, dot.new_zeros(())) / norm
    return [a - coefficient * b for a, b in zip(diversity, smoothness)], bool(dot < 0)
