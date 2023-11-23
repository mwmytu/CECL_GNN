# -*- coding: utf-8 -*-

r"""
NCL
################################################

Reference:
    Zihan Lin*, Changxin Tian*, Yupeng Hou*, Wayne Xin Zhao. "Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning." in WWW 2022.
"""
import collections
import os
import random

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from karateclub import EgoNetSplitter, SCD

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
import numpy as np
from sklearn.cluster import KMeans

def local_cache(cache_name, func):
    # if os.path.exists(f"{cache_name}.pth"):
    #     return torch.load(f"{cache_name}.pth")
    res = func()
    # torch.save(res, f"{cache_name}.pth")
    return res


# 将增加的新功能代码以及被装饰函数运行代码func()一同打包返回，返回的是一个内部函数，这个被返回的函数就是装饰器
class ConstrastComponent(GeneralRecommender):
    r"""NCL is a neighborhood-enriched contrastive learning paradigm for graph collaborative filtering.
    Both structural and semantic neighbors are explicitly captured as contrastive learning objects.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, test_func=None):
        super(ConstrastComponent, self).__init__(config, dataset)
        self.dataset = config['dataset']
        self.addc = config['addc']
        self.step = 0
        self.alpha = config['alpha']
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of lightGCN
        self.dataset = config['dataset']
        self.class_dict = {}
        self.cur_class = 0
        self.user_neighbor_matrix = None
        self.item_neighbor_matrix = None
        self.center_user_embedding = None
        self.center_item_embedding = None
        self.num_clusters = config["num_clusters"]
        self.ssl_temp = config["ssl_temp"]
        self.every_fix_steps = config["every_fix_steps"]
        self.contrast_start_step = config["contrast_start_step"]
        self.reg_weight = config[
            "reg_weight"
        ]
        self.proto_reg = config["proto_reg"]
        self.user_classes, self.item_classes, self.classes = self.acquire_classes()
        self.create_neighbor_matrix()
        print(f"class:{self.classes}")
        self.class_weight = config["class_weight"]
        self.mlp1 = torch.nn.Linear(self.latent_dim, self.classes, bias=False)
        self.mlp2 = torch.nn.Linear(self.latent_dim, self.classes, bias=False)
        self.user_criterion = torch.nn.CrossEntropyLoss()
        self.item_criterion = torch.nn.CrossEntropyLoss()

    def acquire_neighbor_matrix_by_class(self, classes, count):
        # 1. 根据用户的分类构造同类矩阵u*u i*i
        class_dict = collections.defaultdict(list)
        # 1.1 建立[分类->用户]、[分类->物品]字典
        for i in range(len(classes)):
            clazz = classes[i]
            class_dict[clazz].append(i)

        # 1.2 根据字典构建稀疏矩阵
        neighbor_matrix = sp.dok_matrix(
            (count, count), dtype=np.float32
        )
        for i in range(count):
            clazz = classes[i]
            for j in class_dict[clazz]:
                neighbor_matrix[i, j] = 1
        return self.convert_coo_matrix(neighbor_matrix).to(self.device)

    def create_neighbor_matrix(self):
        uc, ic = self.user_classes.cpu().numpy().tolist(), self.item_classes.cpu().numpy().tolist()
        self.user_neighbor_matrix = local_cache(self.dataset + "-user_neighbor_matrix",
                                                lambda: self.acquire_neighbor_matrix_by_class(uc, self.n_users))
        self.item_neighbor_matrix = local_cache(self.dataset + "-item_neighbor_matrix",
                                                lambda: self.acquire_neighbor_matrix_by_class(ic, self.n_items))

    def convert_coo_matrix(self, A):
        # 做个归一化
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        self.diag = torch.from_numpy(diag).to(self.device)
        D = sp.diags(diag)
        L = D @ A @ D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def convert_classes(self, tensor):
        list_res = tensor.cpu().numpy().tolist()
        for i in range(len(list_res)):
            if list_res[i] not in self.class_dict:
                self.class_dict[list_res[i]] = self.cur_class
                self.cur_class += 1
            list_res[i] = self.class_dict[list_res[i]]
        return torch.tensor(list_res, dtype=torch.long).to(self.device)

    def class_loss(self, current_embedding, user, item):
        # 轮流固定embedding
        if self.every_fix_steps and (self.step / self.every_fix_steps) % 2 == 1:
            current_embedding = current_embedding.detach()
        current_user_embeddings, current_item_embeddings = torch.split(
            current_embedding, [self.n_users, self.n_items]
        )
        return self.class_weight * (
                self.user_criterion(self.mlp1(current_user_embeddings[user]), self.user_classes[user]) + \
                self.alpha * self.item_criterion(self.mlp2(current_item_embeddings[item]), self.user_classes[item]))

    def acquire_classes(self):
        if os.path.exists(f"{self.dataset}-user.pth"):
            user_classes = self.convert_classes(torch.load(f"{self.dataset}-user.pth"))
            item_classes = self.convert_classes(torch.load(f"{self.dataset}-item.pth"))
            return user_classes, item_classes, len(
                set(user_classes.cpu().numpy().tolist() + item_classes.cpu().numpy().tolist()))

        G = nx.Graph()
        for i in range(self.interaction_matrix.getnnz()):
            G.add_edge(self.interaction_matrix.col[i] + self.n_users, self.interaction_matrix.row[i])
        for i in range(self.n_items + self.n_users):
            G.add_edge(i, i)
        # splitter = SCD()
        #
        # splitter.fit(G)
        # member_ship = splitter.get_memberships()
        classes_set = set()

        # 构建一个图
        # G = nx.karate_club_graph()

        # 计算拉普拉斯矩阵
        L = nx.laplacian_matrix(G).todense()

        # 计算拉普拉斯矩阵的特征向量和特征值
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        # 使用前两个特征向量进行聚类
        k = self.num_clusters
        X = eigenvectors[:, :k]
        kmeans = KMeans(n_clusters=k).fit(np.asarray(X))
        member_ship = kmeans.labels_
        # member_ship = nx.algorithms.cluster.clustering(G, k=self.num_clusters)
        user_classes = [member_ship[i] for i in range(self.n_users)]
        item_classes = [member_ship[self.n_users + i] for i in range(self.n_items)]
        # user_classes = [random.randint(0,1000) for i in range(self.n_users)]
        # item_classes = [random.randint(0,1000) for i in range(self.n_items)]
        for i in user_classes:
            classes_set.add(i)
        for i in item_classes:
            classes_set.add(i)
        # torch.save(torch.tensor(user_classes, dtype=torch.long).to(torch.device('cpu')), f"{self.dataset}-user.pth")
        # torch.save(torch.tensor(item_classes, dtype=torch.long).to(torch.device('cpu')), f"{self.dataset}-item.pth")
        return self.convert_classes(torch.tensor(user_classes, dtype=torch.long)), self.convert_classes(
            torch.tensor(item_classes, dtype=torch.long)), len(classes_set)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def ProtoNCE_loss(self, node_embedding, user, item):
        # 当训练不充分时不进行对比
        # if self.contrast_start_step < 0 or self.step < self.contrast_start_step:
        #     return torch.tensor(0)
        user_embeddings_all, item_embeddings_all = torch.split(
            node_embedding, [self.n_users, self.n_items]
        )

        user_embeddings = user_embeddings_all[user]  # [B, e]
        norm_user_embeddings = F.normalize(user_embeddings)
        user2centroids = self.center_user_embedding[user]  # [B, e]
        pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.matmul(
            norm_user_embeddings, self.center_user_embedding.transpose(0, 1)
        )
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        item_embeddings = item_embeddings_all[item]
        norm_item_embeddings = F.normalize(item_embeddings)

        item2centroids = self.center_item_embedding[item]  # [B, e]
        pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.matmul(
            norm_item_embeddings, self.center_item_embedding.transpose(0, 1)
        )
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        proto_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + self.alpha * proto_nce_loss_item)
        return proto_nce_loss

    def ssl_layer_loss(self, current_embedding, previous_embedding, user, item):
        current_user_embeddings, current_item_embeddings = torch.split(
            current_embedding, [self.n_users, self.n_items]
        )
        previous_user_embeddings_all, previous_item_embeddings_all = torch.split(
            previous_embedding, [self.n_users, self.n_items]
        )

        current_user_embeddings = current_user_embeddings[user]
        previous_user_embeddings = previous_user_embeddings_all[user]
        norm_user_emb1 = F.normalize(current_user_embeddings)
        norm_user_emb2 = F.normalize(previous_user_embeddings)
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        current_item_embeddings = current_item_embeddings[item]
        previous_item_embeddings = previous_item_embeddings_all[item]
        norm_item_emb1 = F.normalize(current_item_embeddings)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(previous_item_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    def get_center_embedding(self):
        self.center_user_embedding = torch.sparse.mm(self.user_neighbor_matrix,
                                                     F.normalize(self.user_embedding.weight))
        self.center_item_embedding = torch.sparse.mm(self.item_neighbor_matrix,
                                                     F.normalize(self.item_embedding.weight))

    def train(self, mode: bool = True):
        r"""Override train method of base class.The subgraph is reconstructed each time it is called."""
        T = super().train(mode=mode)
        # 每十轮重新生成子图
        if mode:
            self.step += 1
            # self.get_center_embedding()
        return T


if __name__ == '__main__':
    config = collections.defaultdict(str)
    config['dataset'] = "test"


    class Dataset:
        def inter_matrix(self, form):
            # 5个用户5个物品
            ret = sp.dok_matrix(
                (5, 5), dtype=np.float32
            )
            arr = [(0, 0), (0, 2), (0, 4), (1, 0), (1, 2), (1, 4), (2, 1), (2, 4), (3, 4), (3, 1), (4, 0), (4, 4)]
            for (i, j) in arr:
                ret[i, j] = 1
            L = sp.coo_matrix(ret)
            return L


    def test_func(c):
        c.n_users = 5
        c.n_items = 5
        c.device = "cpu"


    ConstrastComponent(config, Dataset(), test_func)
