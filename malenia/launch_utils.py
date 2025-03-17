import os
from dataclasses import dataclass


@dataclass
class CondorParams:
    batch_name: str
    requirements: str  # Ej. '(Machine == "server.com")'
    getenv: bool = True
    should_transfer_files: str = "NO"
    request_CPUs: int = 0
    request_GPUs: int = 0
    request_memory: str = "1G"


def get_datasets_from_types_paths_dict(data_types_and_paths, limited_databases, excluded_databases):
    from malenia.dataset import Dataset

    datasets = []
    for data_type, data_path in data_types_and_paths.items():
        databases = os.listdir(data_path)
        databases.sort()
        for db in databases:
            if limited_databases == "use_all":
                if db not in excluded_databases:
                    datasets.append(Dataset(name=db, dataset_type=data_type, path=data_path))
            else:
                if db in limited_databases:
                    datasets.append(Dataset(name=db, dataset_type=data_type, path=data_path))

    return datasets


def get_datasets_from_path(datasets_path, limited_databases, excluded_databases, remove_dotcsv=False):
    from malenia.dataset import Dataset

    datasets = []
    databases = os.listdir(datasets_path)
    databases.sort()
    for db in databases:
        if remove_dotcsv:
            db = db.replace(".csv", "")
        if limited_databases == "use_all":
            if db not in excluded_databases:
                datasets.append(Dataset(name=db, dataset_type="unknown", path=datasets_path))
        else:
            if db in limited_databases:
                datasets.append(Dataset(name=db, dataset_type="unknown", path=datasets_path))

    return datasets
