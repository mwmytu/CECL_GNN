# -*- coding: utf-8 -*-
# @Time   : 2020/8/31
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

# UPDATE:
# @Time   : 2020/9/16, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com

r"""
LightGCN
################################################

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN
"""
import collections
import os

import networkx as nx
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import torch
from karateclub import EdMot, SCD

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class LightGCN(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)

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
        self.ssl_temp = config["ssl_temp"]

        self.proto_reg = config["proto_reg"]
        self.user_classes, self.item_classes, self.classes = self.acquire_classes()
        self.create_neighbor_matrix()
        print(f"class:{self.classes}")
        self.class_weight = config["class_weight"]
        self.mlp1 = torch.nn.Linear(self.latent_dim, self.classes, bias=False)
        self.mlp2 = torch.nn.Linear(self.latent_dim, self.classes, bias=False)
        self.user_criterion = torch.nn.CrossEntropyLoss()
        self.item_criterion = torch.nn.CrossEntropyLoss()

        self.n_layers = config["n_layers"]  # int type:the layer num of lightGCN
        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def create_neighbor_matrix(self):
        uc, ic = self.user_classes, self.item_classes
        # 1. 根据用户的分类构造同类矩阵u*u i*i
        u_class_dict, i_class_dict = collections.defaultdict(list), collections.defaultdict(list)
        # 1.1 建立[分类->用户]、[分类->物品]字典
        for i in range(uc.shape[0]):
            clazz = uc[i]
            u_class_dict[clazz].append(i)
        for i in range(ic.shape[0]):
            clazz = ic[i]
            i_class_dict[clazz].append(i)

        # 1.2 根据字典构建稀疏矩阵
        user_neighbor_matrix = sp.dok_matrix(
            (self.n_users, self.n_users), dtype=np.float32
        )
        item_neighbor_matrix = sp.dok_matrix(
            (self.n_items, self.n_items), dtype=np.float32
        )
        for i in range(self.n_users):
            clazz = self.user_classes[i]
            for j in u_class_dict[clazz]:
                user_neighbor_matrix[i, j] = 1
        for i in range(self.n_items):
            clazz = self.item_classes[i]
            for j in i_class_dict[clazz]:
                item_neighbor_matrix[i, j] = 1
        self.user_neighbor_matrix = self.convert_coo_matrix(user_neighbor_matrix).to(self.device)
        self.item_neighbor_matrix = self.convert_coo_matrix(item_neighbor_matrix).to(self.device)

    def convert_coo_matrix(self, A):
        L = sp.coo_matrix(A)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def convert_classes(self, tensor):
        tensor = tensor.cpu().numpy().tolist()
        for i in range(len(tensor)):
            if tensor[i] not in self.class_dict:
                self.class_dict[tensor[i]] = self.cur_class
                self.cur_class += 1
            tensor[i] = self.class_dict[tensor[i]]
        return torch.tensor(tensor, dtype=torch.long).to(self.device)

    def class_loss(self, current_embedding, user, item):
        current_user_embeddings, current_item_embeddings = torch.split(
            current_embedding, [self.n_users, self.n_items]
        )
        return self.class_weight * (
                self.user_criterion(self.mlp1(current_user_embeddings[user]), self.user_classes[user]) + \
                self.item_criterion(self.mlp2(current_item_embeddings[item]), self.user_classes[item]))

    def ProtoNCE_loss(self, node_embedding, user, item):
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

        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

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
        splitter = SCD()

        splitter.fit(G)
        member_ship = splitter.get_memberships()
        classes_set = set()

        user_classes = [member_ship[i] for i in range(self.n_users)]
        item_classes = [member_ship[self.n_users + i] for i in range(self.n_items)]
        # user_classes = [random.randint(0,1000) for i in range(self.n_users)]
        # item_classes = [random.randint(0,1000) for i in range(self.n_items)]
        for i in user_classes:
            classes_set.add(i)
        for i in item_classes:
            classes_set.add(i)
        torch.save(torch.tensor(user_classes, dtype=torch.long).to(torch.device('cpu')), f"{self.dataset}-user.pth")
        torch.save(torch.tensor(item_classes, dtype=torch.long).to(torch.device('cpu')), f"{self.dataset}-item.pth")
        return self.convert_classes(torch.tensor(user_classes, dtype=torch.long)), self.convert_classes(
            torch.tensor(item_classes, dtype=torch.long)), len(classes_set)

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        loss = mf_loss + self.reg_weight * reg_loss

        return loss, self.class_loss(torch.cat([user_all_embeddings, item_all_embeddings], dim=0), user, pos_item),\
               self.ProtoNCE_loss(torch.cat([user_all_embeddings, item_all_embeddings], dim=0), user, pos_item)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)

    def train(self, mode: bool = True):
        r"""Override train method of base class.The subgraph is reconstructed each time it is called."""
        T = super().train(mode=mode)
        # 每十轮重新生成子图
        if mode:
            self.center_user_embedding = F.normalize(torch.sparse.mm(self.user_neighbor_matrix, self.user_embedding.weight.detach()))
            self.center_item_embedding = F.normalize(torch.sparse.mm(self.item_neighbor_matrix, self.item_embedding.weight.detach()))
        return T

