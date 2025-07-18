import torch
def kmeans_plusplus(features, n_clusters):
    # features (B, N, D)
    centroids = features[:, -1, :].unsqueeze(1)  # (B, 1, D)
    for i in range(n_clusters - 1):
        features_ex = features.unsqueeze(1).expand(-1, i + 1, -1, -1)  # (B, N, D) -> (B, i, N, D)
        dis = torch.sqrt(torch.sum((features_ex - centroids.unsqueeze(2)) ** 2, dim=-1))  # (B, C, N, D) -> (B, C, N)
        new_centroid_id = torch.argmax(torch.min(dis, dim=1).values, dim=-1)  # (B, C, N) -> (B, N) -> (B)
        new_centroid = torch.gather(features, 1, new_centroid_id.unsqueeze(-1).unsqueeze(-1) \
                                    .expand(-1, -1, features.size(-1)))  # (B, 1, D)
        centroids = torch.cat([centroids, new_centroid], dim=1)
    return centroids


def kmeans(features, n_clusters=2, max_iter=50, device='cuda'):
    features = features.to(device)  # features (B, N, D)
    centroids = kmeans_plusplus(features, n_clusters)  # (B, C, D)
    features_ex = features.unsqueeze(1).expand(-1, n_clusters, -1, -1)  # (B, N, D) -> (B, C, N, D)
    cluster_label = torch.tensor(0)
    label_matrix = torch.tensor(0)
    converged = False
    for i in range(max_iter):
        pre_centroids = centroids
        dis = torch.sqrt(torch.sum((features_ex - centroids.unsqueeze(2)) ** 2, dim=-1))  # (B, C, N, D) -> (B, C, N)
        cluster_label = torch.argmin(dis, dim=1)  # (B, C, N) -> (B, N)
        label_matrix = torch.zeros(cluster_label.size(0), n_clusters, cluster_label.size(-1)).to(device)  # (B, C, N)
        label_matrix.scatter_(1, cluster_label.unsqueeze(1), 1)
        label_sum = torch.sum(label_matrix, dim=-1).unsqueeze(-1)  # (B, C, N) -> (B, C, 1)
        label_matrix /= label_sum
        centroids = torch.bmm(label_matrix, features)  # (B, C, N)*(B, N, D)*  -> (B, C, D)
        if torch.allclose(pre_centroids, centroids):
            converged = True
            break
    if not converged:
        return False ,cluster_label, label_matrix, centroids
        # print('Warning: Clustering is not converged.')
    else:
        return True ,cluster_label, label_matrix, centroids