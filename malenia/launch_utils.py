import os
from dataclasses import dataclass

from malenia.dataset import Dataset


@dataclass
class CondorParams:
    batch_name: str
    requirements: str  # Ej. '(Machine == "server.com")'
    getenv: bool = True
    should_transfer_files: str = "NO"
    request_CPUs: int = 0
    request_GPUs: int = 0
    request_memory: str = "1G"


def get_datasets(data_types_and_paths, limited_databases, excluded_databases):
    datasets = []
    for data_type, data_path in data_types_and_paths.items():
        databases = os.listdir(data_path)
        databases.sort()
        for db in databases if limited_databases == "use_all" else limited_databases:
            if db not in excluded_databases:
                datasets.append(Dataset(name=db, dataset_type=data_type, path=data_path))

    return datasets
