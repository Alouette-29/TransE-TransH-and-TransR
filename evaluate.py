import os
from typing import Union
import torch
from model import TransE, TransH, TransR
from data_processing.utils import read_triples, join_path


script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


def accuracy(predictions: torch.LongTensor, labels: torch.LongTensor) -> float:
    correct = (predictions == labels).sum().item()
    return correct / len(labels)


def evaluate(
        model: Union[TransE, TransH, TransR], 
        triples: list[tuple[int]],
        threshold: float=7) -> float:
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()
    with torch.no_grad():
        all_triples = torch.LongTensor(triples).cuda() if torch.cuda.is_available() else torch.LongTensor(triples)
        h, t, r = [chunk.squeeze(1) for chunk in all_triples.chunk(3, dim=1)]

        if isinstance(model, TransE):
            h_emb = model.embeddings(h).cuda() if torch.cuda.is_available() else model.embeddings(h)
            t_emb = model.embeddings(t).cuda() if torch.cuda.is_available() else model.embeddings(t)
            r_emb = model.embeddings(r).cuda() if torch.cuda.is_available() else model.embeddings(r)

        elif isinstance(model, TransH):
            h_emb = model.entity_embeddings(h).cuda() if torch.cuda.is_available() else model.entity_embeddings(h)
            t_emb = model.entity_embeddings(t).cuda() if torch.cuda.is_available() else model.entity_embeddings(t)
            r_emb = model.relation_embeddings(r).cuda() if torch.cuda.is_available() else model.relation_embeddings(r)
            w_r_emb = model.normal_vector_embeddings(r).cuda() if torch.cuda.is_available() else model.normal_vector_embeddings(r)

            h_emb = h_emb - torch.sum(h_emb * w_r_emb, dim=1, keepdim=True) * w_r_emb
            t_emb = t_emb - torch.sum(t_emb * w_r_emb, dim=1, keepdim=True) * w_r_emb

        elif isinstance(model, TransR):
            h_emb = model.entity_embeddings(h).cuda() if torch.cuda.is_available() else model.entity_embeddings(h)
            t_emb = model.entity_embeddings(t).cuda() if torch.cuda.is_available() else model.entity_embeddings(t)
            r_emb = model.relation_embeddings(r).cuda() if torch.cuda.is_available() else model.relation_embeddings(r)

            h_emb = torch.matmul(h_emb.unsqueeze(1), model.proj_matrix[r]).squeeze(1).cuda()
            t_emb = torch.matmul(t_emb.unsqueeze(1), model.proj_matrix[r]).squeeze(1).cuda()

        distances = torch.norm(h_emb + r_emb - t_emb, p=2, dim=1)

        # 假设距离小于某个阈值的为正确预测
        predictions = (distances < threshold).long().cuda() if torch.cuda.is_available() else (distances < threshold).long()
        labels = torch.ones(len(triples)).long().cuda() if torch.cuda.is_available() else torch.ones(len(triples)).long()
        return accuracy(predictions, labels)
    
    

if __name__ == "__main__":
    valid_triples = read_triples(join_path("valid2id.txt"))
    test_triples = read_triples(join_path("test2id.txt"))
    test_1_1_triples = read_triples(join_path("1-1.txt"))
    test_1_n_triples = read_triples(join_path("1-n.txt"))
    test_n_1_triples = read_triples(join_path("n-1.txt"))
    test_n_n_triples = read_triples(join_path("n-n.txt"))

    num_entities = 14541
    num_relations = 237
    embedding_dim = 64

    modelE = TransE(num_entities, num_relations, embedding_dim).cuda()
    modelH = TransH(num_entities, num_relations, embedding_dim).cuda()
    modelR = TransR(num_entities, num_relations, embedding_dim).cuda()

    modelE.load_state_dict(torch.load(join_path("modelE.pth", "model"), map_location='cuda:0'))
    modelH.load_state_dict(torch.load(join_path("modelH.pth", "model"), map_location='cuda:0'))
    modelR.load_state_dict(torch.load(join_path("modelR.pth", "model"), map_location='cuda:0'))

    if torch.cuda.is_available():
        modelE = modelE.cuda()
        modelH = modelH.cuda()
        modelR = modelR.cuda()

    print("-------------- TransE Model --------------")
    print("Validation Set Accuracy:", evaluate(modelE, valid_triples))
    print("Test Set Accuracy:", evaluate(modelE, test_triples))
    print("1-1 Test Set Accuracy:", evaluate(modelE, test_1_1_triples))
    print("1-n Test Set Accuracy:", evaluate(modelE, test_1_n_triples))
    print("n-1 Test Set Accuracy:", evaluate(modelE, test_n_1_triples))
    print("n-n Test Set Accuracy:", evaluate(modelE, test_n_n_triples))

    print("-------------- TransH Model --------------")
    print("Validation Set Accuracy:", evaluate(modelH, valid_triples))
    print("Test Set Accuracy:", evaluate(modelH, test_triples))
    print("1-1 Test Set Accuracy:", evaluate(modelH, test_1_1_triples))
    print("1-n Test Set Accuracy:", evaluate(modelH, test_1_n_triples))
    print("n-1 Test Set Accuracy:", evaluate(modelH, test_n_1_triples))
    print("n-n Test Set Accuracy:", evaluate(modelH, test_n_n_triples))

    print("-------------- TransR Model --------------")
    print("Validation Set Accuracy:", evaluate(modelR, valid_triples))
    print("Test Set Accuracy:", evaluate(modelR, test_triples))
    print("1-1 Test Set Accuracy:", evaluate(modelR, test_1_1_triples))
    print("1-n Test Set Accuracy:", evaluate(modelR, test_1_n_triples))
    print("n-1 Test Set Accuracy:", evaluate(modelR, test_n_1_triples))
    print("n-n Test Set Accuracy:", evaluate(modelR, test_n_n_triples))