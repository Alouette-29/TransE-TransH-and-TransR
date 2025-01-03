import os
import numpy as np
from tqdm import tqdm
from utils import join_path, read_triples


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)


def generate_negative_samples(
        triples: list[tuple[int]], 
        num_neg_samples: int=3) -> list[tuple[int]]:
    """
    训练集的每一条数据都生成3条负例
    """
    negative_samples = []
    for triple in tqdm(triples):
        h, t, r = triple
        for _ in range(num_neg_samples):
            # 随机选择替换头实体还是尾实体
            mode = np.random.choice(['head', 'tail'])
            if mode == 'head':
                new_h = np.random.choice([i for i in range(num_entities) if i!= h])
                negative_samples.append((new_h, t, r))
            else:
                new_t = np.random.choice([i for i in range(num_entities) if i!= t])
                negative_samples.append((h, new_t, r))
    return negative_samples


if __name__ == "__main__":
    SEED = 42
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)

    num_entities = 14541
    train_triples = read_triples(join_path("train2id.txt"))
    train_negative_samples = generate_negative_samples(train_triples)
    with open(join_path("negative.txt"), "w") as f:
        f.write(f"{len(train_negative_samples)}")
        for triplet in train_negative_samples:
            f.write(" ".join(map(str, triplet)) + "\n")
