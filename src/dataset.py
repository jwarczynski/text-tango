import json
import random
from pathlib import Path
from typing import List, Dict, Any

from torch.utils.data import Dataset

from text_preprocessing import extract_relations


class WebNLGDataset(Dataset):
    def __init__(self, data_dir: str, split: str, samples_per_relation_set: int):
        self.__data_dir = data_dir
        self.__split = split
        self.__data = self.load_data()
        self.__data_keys = list(self.__data.keys())
        self.__samples_per_relation_set = samples_per_relation_set

    def load_data(self):
        train_path = Path(self.__data_dir) / 'train.json'
        val_path = Path(self.__data_dir) / 'dev.json'
        test_path = Path(self.__data_dir) / 'test.json'

        if self.__split == 'train':
            data_path = train_path
        elif self.__split == 'val':
            data_path = val_path
        elif self.__split == 'test':
            data_path = test_path
        else:
            raise ValueError('Invalid split name. Use train, val, or test')

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return preprocess_dataset_file(data)

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx: int | slice) -> tuple[tuple[str, str, str], list[dict[str, str]]] | tuple[tuple, Any]:
        if isinstance(idx, int):  # Single index
            key = self.__data_keys[idx]
            samples = self.__data[key]

            if self.__samples_per_relation_set >= len(samples):
                return key, samples

            random_samples = random.sample(samples, self.__samples_per_relation_set)
            return key, random_samples

        elif isinstance(idx, slice):  # Slice
            start, stop, step = idx.indices(len(self.__data_keys))
            sliced_keys = self.__data_keys[idx]
            sliced_samples = [self.__data[key] for key in sliced_keys]

            # If step is 1, return list of tuples (keys, samples) for each index
            if step == 1:
                return [(key, self.__data[key]) for key in sliced_keys]

            # Otherwise, return a single tuple (keys, samples)
            return sliced_keys, sliced_samples

def preprocess_dataset_file(json_data: Dict[str, List[Dict[str, str]]]) -> dict[tuple, list[dict[str, str]]]:
    data_dict = {}
    for i in range(len(json_data['data'])):
        sample = json_data['data'][i]
        in_data = json_data['data'][i]['in']
        relations = extract_relations(in_data)
        relations = tuple(relations)

        if relations in data_dict:
            data_dict[relations].append(sample)
        else:
            data_dict[relations] = [sample]

    return data_dict
