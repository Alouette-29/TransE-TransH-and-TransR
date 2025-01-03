import torch
import torch.nn as nn


class TransE(nn.Module):
    """
    实体e和关系r都在同一个向量空间里
    运算就在原空间
    只需要学习embedding
    """
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int):
        super(TransE, self).__init__()
        self.embeddings = nn.Embedding(num_entities + num_relations, embedding_dim)
        self.margin = 1.0
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, positive_triples: torch.LongTensor, negative_triples: torch.LongTensor) -> torch.Tensor:
        """
        正负样本都是形如(batch_size, 3)的一组三元组
        """
        h_pos, t_pos, r_pos = [chunk.squeeze(1) for chunk in positive_triples.chunk(3, dim=1)]  # (batch_size,)
        h_neg, t_neg, r_neg = [chunk.squeeze(1) for chunk in negative_triples.chunk(3, dim=1)]

        h_pos_emb = self.embeddings(h_pos)
        t_pos_emb = self.embeddings(t_pos)
        r_pos_emb = self.embeddings(r_pos)

        h_neg_emb = self.embeddings(h_neg)
        t_neg_emb = self.embeddings(t_neg)
        r_neg_emb = self.embeddings(r_neg)

        pos_distance = torch.norm(h_pos_emb + r_pos_emb - t_pos_emb, p=2, dim=1)
        neg_distance = torch.norm(h_neg_emb + r_neg_emb - t_neg_emb, p=2, dim=1)

        loss = torch.sum(torch.relu(self.margin + pos_distance - neg_distance))
        return loss
    

class TransH(nn.Module):
    """
    实体e和关系r仍在同一空间
    运算时实体e投影到关系r对应的一个超平面上
    使得遇到不同的r 同一个e也可能有不同的投影表示
    每个r都需要学习一个超平面的法向量
    """
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int):
        super(TransH, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.normal_vector_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = 1.0
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        nn.init.xavier_uniform_(self.normal_vector_embeddings.weight)

    def forward(self, positive_triples: torch.LongTensor, negative_triples: torch.LongTensor) -> torch.Tensor:
        h_pos, t_pos, r_pos = [chunk.squeeze(1) for chunk in positive_triples.chunk(3, dim=1)]
        h_neg, t_neg, r_neg = [chunk.squeeze(1) for chunk in negative_triples.chunk(3, dim=1)]

        h_pos_emb = self.entity_embeddings(h_pos)
        t_pos_emb = self.entity_embeddings(t_pos)
        r_pos_emb = self.relation_embeddings(r_pos)
        w_r_pos_emb = self.normal_vector_embeddings(r_pos)

        h_neg_emb = self.entity_embeddings(h_neg)
        t_neg_emb = self.entity_embeddings(t_neg)
        r_neg_emb = self.relation_embeddings(r_neg)
        w_r_neg_emb = self.normal_vector_embeddings(r_neg)

        h_pos_proj = h_pos_emb - torch.sum(h_pos_emb * w_r_pos_emb, dim=1, keepdim=True) * w_r_pos_emb
        t_pos_proj = t_pos_emb - torch.sum(t_pos_emb * w_r_pos_emb, dim=1, keepdim=True) * w_r_pos_emb

        h_neg_proj = h_neg_emb - torch.sum(h_neg_emb * w_r_neg_emb, dim=1, keepdim=True) * w_r_neg_emb
        t_neg_proj = t_neg_emb - torch.sum(t_neg_emb * w_r_neg_emb, dim=1, keepdim=True) * w_r_neg_emb

        pos_distance = torch.norm(h_pos_proj + r_pos_emb - t_pos_proj, p=2, dim=1)
        neg_distance = torch.norm(h_neg_proj + r_neg_emb - t_neg_proj, p=2, dim=1)
        loss = torch.sum(torch.relu(self.margin + pos_distance - neg_distance))

        return loss


class TransR(nn.Module):
    """
    实体和关系分别在不同的空间
    运算时实体e投影到关系r空间
    每个r都需要学习一个映射矩阵 把h和t映射到r的空间
    """
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int):
        super(TransR, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.proj_matrix = nn.Parameter(torch.Tensor(num_relations, embedding_dim, embedding_dim))
        self.margin = 1.0
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        nn.init.xavier_uniform_(self.proj_matrix)

    def forward(self, positive_triples: torch.LongTensor, negative_triples: torch.LongTensor) -> torch.Tensor:
        h_pos, t_pos, r_pos = [chunk.squeeze(1) for chunk in positive_triples.chunk(3, dim=1)]
        h_neg, t_neg, r_neg = [chunk.squeeze(1) for chunk in negative_triples.chunk(3, dim=1)]

        h_pos_emb = self.entity_embeddings(h_pos)
        t_pos_emb = self.entity_embeddings(t_pos)
        r_pos_emb = self.relation_embeddings(r_pos)

        h_neg_emb = self.entity_embeddings(h_neg)
        t_neg_emb = self.entity_embeddings(t_neg)
        r_neg_emb = self.relation_embeddings(r_neg)

        h_pos_proj = torch.matmul(h_pos_emb.unsqueeze(1), self.proj_matrix[r_pos]).squeeze(1)
        t_pos_proj = torch.matmul(t_pos_emb.unsqueeze(1), self.proj_matrix[r_pos]).squeeze(1)

        h_neg_proj = torch.matmul(h_neg_emb.unsqueeze(1), self.proj_matrix[r_neg]).squeeze(1)
        t_neg_proj = torch.matmul(t_neg_emb.unsqueeze(1), self.proj_matrix[r_neg]).squeeze(1)

        pos_distance = torch.norm(h_pos_proj + r_pos_emb - t_pos_proj, p=2, dim=1)
        neg_distance = torch.norm(h_neg_proj + r_neg_emb - t_neg_proj, p=2, dim=1)
        loss = torch.sum(torch.relu(self.margin + pos_distance - neg_distance))

        return loss