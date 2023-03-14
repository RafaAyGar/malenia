from dataclasses import dataclass

@dataclass
class CondorParams:
    batch_name : str
    requirements : str # Ej. '(Machine == "server.com")'
    getenv : bool = True
    should_transfer_files : str = "NO"
    request_CPUs : int = 0
    request_GPUs : int = 0
    request_memory : str = "1G"