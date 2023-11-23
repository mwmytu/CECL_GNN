# -*- coding: utf-8 -*-
r"""
NCL
################################################
Reference:
    Zihan Lin*, Changxin Tian*, Yupeng Hou*, Wayne Xin Zhao. "Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning." in WWW 2022.
"""

import numpy as np
import scipy.sparse as sp
import torch
import os
import torch.nn.functional as F

import faiss
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class NCL(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(NCL, self).__init__(config, dataset)
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        # 是否开启聚类
        self.cluster = config['cluster']
        # load parameters info
        self.latent_dim = config['embedding_size']  # int type: the embedding size of the base model
        self.n_layers = config['n_layers']  # int type: the layer num of the base model
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization

        self.ssl_temp = config['ssl_temp']
        self.ssl_reg = config['ssl_reg']
        # 这个是偶数跳
        self.hyper_layers = config['hyper_layers']

        self.alpha = config['alpha']

        self.proto_reg = config['proto_reg']
        self.k = config['num_clusters']
        # SGL
        self._user = dataset.inter_feat[dataset.uid_field]
        self._item = dataset.inter_feat[dataset.iid_field]
        self.embed_dim = config["embedding_size"]
        self.type = config["type"]
        self.drop_ratio = config["drop_ratio"]
        self.ssl_tau = config["ssl_tau"]
        self.ssl_weight = config["ssl_weight"]
        self.graph_count = config['graph_count']
        self.types = [0] * 2
        self.types[0] = config["type0"]
        self.types[1] = config["type1"]
        # self.train_graph = self.csr2tensor(self.create_adjust_matrix(is_sub=False))
        self.train_graph = self.create_adjust_matrix(is_sub=False)
        # import recbole
        # neumf_model = run_recbole(model="NeuMF", dataset=config['dataset'],
        #                           config_file_list=['properties/NeuMF.yaml', f'properties/ml-1m.yaml'])
        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim
                                                 )
        # , _weight=neumf_model.user_mf_embedding.weight)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim
                                                 )
        # ,_weight=neumf_model.item_mf_embedding.weight)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # self.norm_adj_mat = self.get_norm_adj_mat().to(self.device)
        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None
        self.sub_graphs = []
        self.proto = False

    def e_step(self):

        self.proto = True
        user_emd, item_emd = self.user_embedding, self.item_embedding
        user_embeddings = user_emd.weight.detach().cpu().numpy()
        item_embeddings = item_emd.weight.detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x
        """
        kmeans = faiss.Kmeans(d=self.latent_dim, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster

    def get_norm_adj_mat(self, inter_M):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M_t = inter_M.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), inter_M.data))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), inter_M_t.data)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        self.diag = torch.from_numpy(diag).to(self.device)
        D = sp.diags(diag)
        L = D @ A @ D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        # (U + I) * (U + I)的对称矩阵,物品对物品、用户对用户是
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

    def forward(self, graph):
        main_ego = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_ego = [main_ego]
        if isinstance(graph, list):
            for sub_graph in graph:
                main_ego = torch.sparse.mm(sub_graph, main_ego)
                all_ego.append(main_ego)
        else:
            for i in range(self.n_layers):
                main_ego = torch.sparse.mm(graph, main_ego)
                all_ego.append(main_ego)
        all_ego = torch.stack(all_ego, dim=1)
        all_ego = torch.mean(all_ego, dim=1, keepdim=False)
        user_emd, item_emd = torch.split(all_ego, [self.n_users, self.n_items], dim=0)

        return user_emd, item_emd

    def forward_second(self, graph):
        main_ego = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_ego = [main_ego]
        if isinstance(graph, list):
            for sub_graph in graph:
                main_ego = torch.sparse.mm(sub_graph, main_ego)
                all_ego.append(main_ego)
        else:
            for i in range(self.n_layers):
                main_ego = torch.sparse.mm(graph, main_ego)
                all_ego.append(main_ego)
        second_user, second_item = torch.split(all_ego[2], [self.n_users, self.n_items], dim=0)
        return second_user, second_item

    def cov(self, all):
        all = torch.stack(all, dim=1)
        all = torch.mean(all, dim=1, keepdim=False)
        user_emd, item_emd = torch.split(all, [self.n_users, self.n_items], dim=0)
        return user_emd, item_emd

    # def forward(self):
    #     all_embeddings = self.get_ego_embeddings()
    #     embeddings_list = [all_embeddings]
    #     # 这里是light gcn!!将邻居信息聚合
    #     for layer_idx in range(max(self.n_layers, self.hyper_layers * 2)):
    #         # 聚合
    #         all_embeddings = torch.sparse.mm(self.norm_adj_mat, all_embeddings)
    #         embeddings_list.append(all_embeddings)
    #     lightgcn_all_embeddings = torch.stack(embeddings_list[:self.n_layers + 1], dim=1)
    #     # gnn特征，通过对所有层的特征求均值合并最终的特征
    #     lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
    #
    #     user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
    #     return user_all_embeddings, item_all_embeddings, embeddings_list

    def ProtoNCE_loss(self, node_embedding, user, item):
        user_embeddings_all, item_embeddings_all = torch.split(node_embedding, [self.n_users, self.n_items])

        user_embeddings = user_embeddings_all[user]  # [B, e]
        norm_user_embeddings = F.normalize(user_embeddings)
        user2cluster = self.user_2cluster[user]  # [B,]
        user2centroids = self.user_centroids[user2cluster]  # [B, e]
        pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.matmul(norm_user_embeddings, self.user_centroids.transpose(0, 1))
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        item_embeddings = item_embeddings_all[item]
        norm_item_embeddings = F.normalize(item_embeddings)

        item2cluster = self.item_2cluster[item]  # [B, ]
        item2centroids = self.item_centroids[item2cluster]  # [B, e]
        pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.matmul(norm_item_embeddings, self.item_centroids.transpose(0, 1))
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        proto_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

    def ssl_layer_loss(self, current_embedding, previous_embedding, user, item):
        current_user_embeddings, current_item_embeddings = torch.split(current_embedding, [self.n_users, self.n_items])
        previous_user_embeddings_all, previous_item_embeddings_all = torch.split(previous_embedding,
                                                                                 [self.n_users, self.n_items])

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

    # def predict(self, interaction):
    #     user = interaction[self.USER_ID]
    #     item = interaction[self.ITEM_ID]
    #
    #     user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()
    #
    #     u_embeddings = user_all_embeddings[user]
    #     i_embeddings = item_all_embeddings[item]
    #     scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
    #     return scores

    def predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward_second(self.train_graph)

        user = self.restore_user_e[interaction[self.USER_ID]]
        item = self.restore_item_e[interaction[self.ITEM_ID]]
        return torch.sum(user * item, dim=1)

    # def full_sort_predict(self, interaction):
    #     user = interaction[self.USER_ID]
    #     if self.restore_user_e is None or self.restore_item_e is None:
    #         self.restore_user_e, self.restore_item_e, embedding_list = self.forward()
    #     # get user embedding from storage variable
    #     u_embeddings = self.restore_user_e[user]
    #     # 画个圆
    #     # dot with all item embedding to accelerate
    #
    #     scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
    #     return scores.view(-1)

    def graph_construction(self):
        self.sub_graphs = []
        for i in range(self.graph_count):
            self.type = self.types[i]
            r"""Devise three operators to generate the views — node dropout, edge dropout, and random walk of a node."""
            sub_graph = []
            if self.type == "ND" or self.type == "ED":
                sub_graph = self.create_adjust_matrix(is_sub=True)
            elif self.type == "RW":
                for i in range(self.n_layers):
                    _g = self.create_adjust_matrix(is_sub=True)
                    sub_graph.append(_g)
            self.sub_graphs.append(sub_graph)

    def rand_sample(self, high, size=None, replace=True):
        r"""Randomly discard some points or edges.

        Args:
            high (int): Upper limit of index value
            size (int): Array size after sampling

        Returns:
            numpy.ndarray: Array index after sampling, shape: [size]
        """

        a = np.arange(high)
        sample = np.random.choice(a, size=size, replace=replace)
        return sample

    def create_adjust_matrix(self, is_sub: bool):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.If it is a subgraph, it may be processed by
        node dropout or edge dropout.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            csr_matrix of the normalized interaction matrix.
        """
        matrix = None
        if not is_sub:
            ratings = np.ones_like(self._user, dtype=np.float32)
            matrix = sp.csr_matrix(
                (ratings, (self._user, self._item)),
                shape=(self.n_users, self.n_items),
            )
        else:
            if self.type == "ND":
                drop_user = self.rand_sample(
                    self.n_users,
                    size=int(self.n_users * self.drop_ratio),
                    replace=False,
                )
                drop_item = self.rand_sample(
                    self.n_items,
                    size=int(self.n_items * self.drop_ratio),
                    replace=False,
                )
                R_user = np.ones(self.n_users, dtype=np.float32)
                R_user[drop_user] = 0.0
                R_item = np.ones(self.n_items, dtype=np.float32)
                R_item[drop_item] = 0.0
                R_user = sp.diags(R_user)
                R_item = sp.diags(R_item)
                R_G = sp.csr_matrix(
                    (
                        np.ones_like(self._user, dtype=np.float32),
                        (self._user, self._item),
                    ),
                    shape=(self.n_users, self.n_items),
                )
                res = R_user.dot(R_G)
                res = res.dot(R_item)

                user, item = res.nonzero()
                ratings = res.data
                matrix = sp.csr_matrix(
                    (ratings, (user, item)),
                    shape=(self.n_users, self.n_items),
                )

            elif self.type == "ED" or self.type == "RW":
                keep_item = self.rand_sample(
                    len(self._user),
                    size=int(len(self._user) * (1 - self.drop_ratio)),
                    replace=False,
                )
                user = self._user[keep_item]
                item = self._item[keep_item]

                matrix = sp.csr_matrix(
                    (np.ones_like(user), (user, item)),
                    shape=(self.n_users,self.n_items),
                )

        # matrix = matrix + matrix.T
        # D = np.array(matrix.sum(axis=1)) + 1e-7
        # D = np.power(D, -0.5).flatten()
        # D = sp.diags(D)

        return self.get_norm_adj_mat(matrix.tocoo()).to(self.device)

    def csr2tensor(self, matrix: sp.csr_matrix):
        r"""Convert csr_matrix to tensor.

        Args:
            matrix (scipy.csr_matrix): Sparse matrix to be converted.

        Returns:
            torch.sparse.FloatTensor: Transformed sparse matrix.
        """
        matrix = matrix.tocoo()
        x = torch.sparse.FloatTensor(
            torch.LongTensor(np.array([matrix.row, matrix.col])),
            torch.FloatTensor(matrix.data.astype(np.float32)),
            matrix.shape,
        ).to(self.device)
        return x

    def calculate_loss1(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        # center_embedding原来是
        ego_embedding = embeddings_list[0]
        # 做了hyper_layer * 2次聚合，从而得到所有聚合后的嵌入信息
        context_embedding = embeddings_list[self.hyper_layers * 2]

        ssl_loss = self.ssl_layer_loss(context_embedding, ego_embedding, user, pos_item)
        proto_loss = self.ProtoNCE_loss(ego_embedding, user, pos_item)

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        return mf_loss + self.reg_weight * reg_loss, ssl_loss, proto_loss

    # def calculate_loss(self, interaction):
    #     if self.restore_user_e is not None or self.restore_item_e is not None:
    #         self.restore_user_e, self.restore_item_e = None, None
    #
    #     user_list = interaction[self.USER_ID]
    #     pos_item_list = interaction[self.ITEM_ID]
    #     neg_item_list = interaction[self.NEG_ITEM_ID]
    #     previous_embedding = self.get_ego_embeddings()
    #     user_emd_second, item_emd_second = self.forward_second(self.train_graph)
    #     user_emd, item_emd = self.forward_second(self.train_graph)
    #     ssl_loss = self.ssl_layer_loss(previous_embedding, torch.cat([user_emd_second, item_emd_second], dim=0),
    #                                    user_list, pos_item_list)
    #     proto_loss = self.ProtoNCE_loss(previous_embedding, user_list, pos_item_list)
    #     user_sub1, item_sub1 = self.forward(self.sub_graphs[0])
    #     user_sub2, item_sub2 = self.forward(self.sub_graphs[1])
    #     bpr_loss = self.calc_bpr_loss(
    #         user_emd, item_emd, user_list, pos_item_list, neg_item_list
    #     )
    #     # u_ego_embeddings = self.user_embedding(user_list)
    #     # pos_ego_embeddings = self.item_embedding(pos_item_list)
    #     # neg_ego_embeddings = self.item_embedding(neg_item_list)
    #
    #     sub_ssl_loss = self.calc_ssl_loss(user_list, pos_item_list, user_sub1, user_sub2, item_sub1, item_sub2)
    #     return bpr_loss, ssl_loss + sub_ssl_loss, proto_loss
    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        user_list = interaction[self.USER_ID]
        pos_item_list = interaction[self.ITEM_ID]
        neg_item_list = interaction[self.NEG_ITEM_ID]
        user_emd, item_emd = self.forward(self.train_graph)
        user_sub1, item_sub1 = self.forward(self.sub_graphs[0])
        user_sub2, item_sub2 = self.forward(self.sub_graphs[1])
        proto_loss = self.ProtoNCE_loss(self.get_ego_embeddings(), user_list, pos_item_list)

        total_loss = self.calc_bpr_loss(
            user_emd, item_emd, user_list, pos_item_list, neg_item_list
        ) + self.calc_ssl_loss(
            user_list, pos_item_list, user_sub1, user_sub2, item_sub1, item_sub2
        ) + proto_loss
        return total_loss
    def calc_bpr_loss(
            self, user_emd, item_emd, user_list, pos_item_list, neg_item_list
    ):
        r"""Calculate the the pairwise Bayesian Personalized Ranking (BPR) loss and parameter regularization loss.

        Args:
            user_emd (torch.Tensor): Ego embedding of all users after forwarding.
            item_emd (torch.Tensor): Ego embedding of all items after forwarding.
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.
            neg_item_list (torch.Tensor): List of negative examples.

        Returns:
            torch.Tensor: Loss of BPR tasks and parameter regularization.
        """
        u_e = user_emd[user_list]
        pi_e = item_emd[pos_item_list]
        ni_e = item_emd[neg_item_list]
        p_scores = torch.mul(u_e, pi_e).sum(dim=1)
        n_scores = torch.mul(u_e, ni_e).sum(dim=1)

        l1 = torch.sum(-F.logsigmoid(p_scores - n_scores))

        u_e_p = self.user_embedding(user_list)
        pi_e_p = self.item_embedding(pos_item_list)
        ni_e_p = self.item_embedding(neg_item_list)

        l2 = self.reg_loss(u_e_p, pi_e_p, ni_e_p)

        return l1 + l2 * self.reg_weight

    def calc_ssl_loss(
            self, user_list, pos_item_list, user_sub1, user_sub2, item_sub1, item_sub2
    ):
        r"""Calculate the loss of self-supervised tasks.

        Args:
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.
            user_sub1 (torch.Tensor): Ego embedding of all users in the first subgraph after forwarding.
            user_sub2 (torch.Tensor): Ego embedding of all users in the second subgraph after forwarding.
            item_sub1 (torch.Tensor): Ego embedding of all items in the first subgraph after forwarding.
            item_sub2 (torch.Tensor): Ego embedding of all items in the second subgraph after forwarding.

        Returns:
            torch.Tensor: Loss of self-supervised tasks.
        """

        u_emd1 = F.normalize(user_sub1[user_list], dim=1)
        u_emd2 = F.normalize(user_sub2[user_list], dim=1)
        all_user2 = F.normalize(user_sub2, dim=1)
        v1 = torch.sum(u_emd1 * u_emd2, dim=1)
        v2 = u_emd1.matmul(all_user2.T)
        v1 = torch.exp(v1 / self.ssl_tau)
        v2 = torch.sum(torch.exp(v2 / self.ssl_tau), dim=1)
        ssl_user = -torch.sum(torch.log(v1 / v2))

        i_emd1 = F.normalize(item_sub1[pos_item_list], dim=1)
        i_emd2 = F.normalize(item_sub2[pos_item_list], dim=1)
        all_item2 = F.normalize(item_sub2, dim=1)
        v3 = torch.sum(i_emd1 * i_emd2, dim=1)
        v4 = i_emd1.matmul(all_item2.T)
        v3 = torch.exp(v3 / self.ssl_tau)
        v4 = torch.sum(torch.exp(v4 / self.ssl_tau), dim=1)
        ssl_item = -torch.sum(torch.log(v3 / v4))

        return (ssl_item + ssl_user) * self.ssl_weight

    def full_sort_predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward_second(self.train_graph)

        user = self.restore_user_e[interaction[self.USER_ID]]
        return user.matmul(self.restore_item_e.T)

    def train(self, mode: bool = True):
        r"""Override train method of base class.The subgraph is reconstructed each time it is called."""
        T = super().train(mode=mode)
        if mode:
            self.graph_construction()
        return T
