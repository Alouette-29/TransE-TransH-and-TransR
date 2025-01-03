import os


DATA_DIR = "FB15K237"


def join_path(filename: str, data_dir: str=DATA_DIR) -> str:
    return os.path.join(data_dir, filename)


def read_triples(file_path: str) -> list[tuple[int]]:
    triples = []
    with open(file_path, 'r') as f:
        for line in f.readlines()[1:]:
            h, t, r = line.strip().split()
            h = int(h)
            t = int(t)
            r = int(r)
            triples.append((h, t, r))
    return triples
