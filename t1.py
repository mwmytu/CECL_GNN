import numpy as np
import scipy.sparse as sp
import torch
from recbole.quick_start import load_data_and_model


class Test:

    def __init__(self):
        self.n_users = 10
        self.n_items = 10
        self.device = "cuda"
        data = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        row = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        col = [1, 3, 7, 2, 4, 8, 3, 4, 8]
        self.interaction_matrix = sp.coo_matrix((data, (row, col)), shape=(10, 10))


def get_norm_adj_mat(my):
    r"""Get the normalized interaction matrix of users and items.

    Construct the square matrix from the training data and normalize it
    using the laplace matrix.

    .. math::
        A_{hat} = D^{-0.5} \times A \times D^{-0.5}

    Returns:
        Sparse tensor of the normalized interaction matrix.
    """
    # build adj matrix
    A = sp.dok_matrix((my.n_users + my.n_items, my.n_users + my.n_items), dtype=np.float32)
    inter_M = my.interaction_matrix
    inter_M_t = my.interaction_matrix.transpose()
    data_dict = dict(zip(zip(inter_M.row, inter_M.col + my.n_users), inter_M.data))
    data_dict.update(dict(zip(zip(inter_M_t.row + my.n_users, inter_M_t.col), inter_M_t.data)))
    A._update(data_dict)
    # norm adj matrix
    sumArr = (A > 0).sum(axis=1)
    # add epsilon to avoid divide by zero Warning
    diag = np.array(sumArr.flatten())[0] + 1e-7
    diag = np.power(diag, -0.5)
    my.diag = torch.from_numpy(diag).to(my.device)
    D = sp.diags(diag)
    L = D @ A @ D
    # covert norm_adj matrix to tensor
    L = sp.coo_matrix(L)
    row = L.row
    col = L.col
    i = torch.LongTensor([row, col])
    data = torch.FloatTensor(L.data)
    SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
    return SparseL

def show_model(data, index, n_clusters,is_user, step):
    import torch
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # Query some points to get their nearest clusters
    _, I = index.search(data, 1)
    clusters = I.reshape(-1)

    # Use t-SNE to reduce the embeddings to 2D
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    embeddings_2d = tsne.fit_transform(data)

    # Define a list of 1000 colors with different hues
    colors = plt.cm.hsv(np.linspace(0, 1, n_clusters))

    # Define a list of 10 markers with different styles
    markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'h', '+', 'x']

    # Plot the 2D embeddings with different colors and markers for each cluster
    for i in range(n_clusters):
        mask = clusters == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], color=colors[i], marker=markers[i % 10])

    # Show the plot
    plt.title(f"{'u' if is_user else 'i'} with step {step}")
    plt.show()
    # import os
    # os.mkdir("imgs")
    # plt.savefig(f'imgs/ncl{step}.png')

# get_norm_adj_mat(Test())
if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from scipy.stats import gaussian_kde
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file='saved/SGL-Apr-06-2023_08-45-13.pth',
    )
    embedding_layer = model.user_embedding.weight  # num_embeddings=1000, embedding_dim=50
    embedding_shape = embedding_layer.weight.shape


    # 使用 TSNE 算法将单词向量降维到 2 维
    tsne = TSNE(n_components=2)
    data_tsne = tsne.fit_transform(embedding_layer)
    # 可视化结果
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1])

    # for i, txt in enumerate(range(10)):
    #     plt.annotate(txt, (data_tsne[i, 0], data_tsne[i, 1]))
    plt.show()
    # # 将t-SNE降维后的数据进行密度估计
    # x = data_tsne[:, 0]
    # y = data_tsne[:, 1]
    # kde = gaussian_kde(np.vstack([x, y]))
    # xx, yy = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    # density = np.reshape(kde(np.vstack([xx.ravel(), yy.ravel()])), xx.shape)
    #
    # # 绘制热图
    # fig, ax = plt.subplots()
    # im = ax.imshow(density, cmap='coolwarm')
    #
    # # 添加颜色条
    # cbar = ax.figure.colorbar(im, ax=ax)
    #
    # # 设置坐标轴标签
    # ax.set_xticks(np.arange(0, 100, 20))
    # ax.set_yticks(np.arange(0, 100, 20))
    # ax.set_xticklabels(np.arange(0, 1000, 200))
    # ax.set_yticklabels(np.arange(0, 1000, 200))
    #
    # # 设置坐标轴标签和标题
    # plt.xlabel('t-SNE1')
    # plt.ylabel('t-SNE2')
    # plt.title('t-SNE Heatmap')
    #
    # # 显示热图
    # plt.show()
