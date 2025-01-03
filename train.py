import os
from typing import Union
import numpy as np
import torch
import torch.optim as optim
from model import TransE, TransH, TransR
from data_processing.utils import read_triples, join_path
import matplotlib.pyplot as plt
from tqdm import tqdm


script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


def seed_everything(TORCH_SEED: int=42):
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
     

seed_everything()


def train(
        model: Union[TransE, TransH, TransR],
        optimizer: torch.optim.Adam,
        num_epochs: int,
        batch_size: int,
        train_triples: list[tuple[int]],
        train_negative_samples: list[tuple[int]],
        model_filename: str,
        model_dir: str="model"):
    
    if not os.path.exists(model_dir):
         os.makedirs(model_dir)
    
    assert len(train_triples) * 3 == len(train_negative_samples)
    train_triples *= 3

    np.random.shuffle(train_triples)
    np.random.shuffle(train_negative_samples)

    if torch.cuda.is_available():
        model = model.cuda()

    losses = []
    
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0

        for i in range(0, len(train_triples), batch_size):
            positive_batch = train_triples[i: i + batch_size]
            negative_batch = train_negative_samples[i: i + batch_size]
            positive_batch = torch.LongTensor(positive_batch).cuda() if torch.cuda.is_available() else torch.LongTensor(positive_batch)
            negative_batch = torch.LongTensor(negative_batch).cuda() if torch.cuda.is_available() else torch.LongTensor(negative_batch)

            optimizer.zero_grad()
            loss = model(positive_batch, negative_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'Epoch {epoch + 1}: Loss = {epoch_loss}')
        losses.append(epoch_loss)

    torch.save(model.state_dict(), join_path(model_filename, model_dir))

    losses = losses[30:]  # 前面loss太大了

    if not os.path.exists('fig'):
        os.makedirs('fig')
    plt.figure()
    plt.plot(losses, label="Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"fig/{model_filename[:-4]}_loss.png")
    plt.close()


if __name__ == "__main__":
    num_entities = 14541
    num_relations = 237
    embedding_dim = 64

    modelE = TransE(num_entities, num_relations, embedding_dim)
    modelH = TransH(num_entities, num_relations, embedding_dim)
    modelR = TransR(num_entities, num_relations, embedding_dim)
    optimizerE = optim.Adam(modelE.parameters(), lr=0.001)
    optimizerH = optim.Adam(modelH.parameters(), lr=0.001)
    optimizerR = optim.Adam(modelR.parameters(), lr=0.001)

    batch_size = 128

    train_triples = read_triples(join_path("train2id.txt"))
    train_negative_samples =  read_triples(join_path("negative.txt"))

    # train(modelE, optimizerE, 150, batch_size, train_triples, train_negative_samples, "modelE.pth")
    # train(modelH, optimizerH, 200, batch_size, train_triples, train_negative_samples, "modelH.pth")
    train(modelR, optimizerR, 250, batch_size, train_triples, train_negative_samples, "modelR.pth")
