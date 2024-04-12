import time
from abc import ABCMeta
from pathlib import Path
from typing import List, Callable

import torch
import torch.nn as nn
from s3torchconnector import S3Checkpoint
from s3torchconnector._s3dataset_common import parse_s3_uri
from torch.utils.data import DataLoader

from .common import Distribution
from .monitor import Measurement


class Scenario(metaclass=ABCMeta):
    def run(self) -> List[Measurement]:
        raise NotImplementedError


class DummyScenario(Scenario):

    def run(self) -> List[Measurement]:
        time.sleep(5)
        return [Measurement(label="volume", value=42.), Measurement(label="elapsed_time", value=1024.)]


class DataLoadingScenario(Scenario):
    def __init__(self, dataloader: DataLoader, epochs: int):
        self.dataloader = dataloader
        self.epochs = epochs

    def run(self) -> List[Measurement]:
        num_samples = 0
        start_time = time.perf_counter()
        for epoch in range(0, self.epochs):
            for batch_idx, (data, target) in enumerate(self.dataloader):
                num_samples += len(data)
        end_time = time.perf_counter()
        training_time = end_time - start_time

        return [
            Measurement(label="elapsed_time", value=training_time),
            Measurement(label="volume", value=num_samples)
        ]


class Checkpointer(metaclass=ABCMeta):

    def checkpoint(self, model: nn.Module, disambiguator: str) -> float:
        start_time = time.perf_counter()
        self.do_checkpoint(model, disambiguator)
        elapsed_time = time.perf_counter() - start_time

        return elapsed_time

    def do_checkpoint(self, model: nn.Module, disambiguator: str):
        raise NotImplementedError


class TorchDiskCheckpointer(Checkpointer):
    def __init__(self, directory: Path = Path(".")):
        self.directory = directory

    def do_checkpoint(self, model: nn.Module, disambiguator: str):
        path = self.directory / f"{disambiguator}.ckpt"
        torch.save(model.state_dict(), path)


class TorchS3Checkpointer(Checkpointer):
    def __init__(self, region: str, s3_prefix: str):
        self.checkpoint_manager = S3Checkpoint(region=region)
        self.bucket, self.key_prefix = parse_s3_uri(s3_prefix)

    def _build_key(self, disambiguator: str) -> str:
        path = Path(self.bucket) / self.key_prefix / f"{disambiguator}.ckpt"
        return "s3://{0}".format(path)

    def do_checkpoint(self, model: nn.Module, disambiguator: str):
        key = self._build_key(disambiguator)
        with self.checkpoint_manager.writer(key) as writer:
            torch.save(model.state_dict(), writer)


class CheckpointingScenario(Scenario):

    def __init__(self, model: nn.Module, repetitions: int, checkpointer: Checkpointer) -> None:
        self.model = model
        self.repetitions = repetitions
        self.checkpointer = checkpointer

    def run(self) -> List[Measurement]:
        save_times = Distribution()
        for i in range(self.repetitions):
            elapsed_time = self.checkpointer.checkpoint(self.model, f"iteration_{i}")
            save_times.add(elapsed_time)

        return [Measurement(label="save_times", value=save_times)]


class TrainingScenario(Scenario):
    def __init__(self, dataloader: DataLoader, epochs: int,
                 model: nn.Module, loss_fn: torch.nn.modules.loss._Loss = None,
                 optimizer: torch.optim.optimizer.Optimizer = None,
                 device=None,
                 checkpoint_criteria: Callable[[int], bool] = None,
                 checkpointer: Checkpointer = None):
        self.dataloader = dataloader
        self.epochs = epochs
        self.model = model
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_criteria = checkpoint_criteria or (lambda _: False)
        self.checkpointer = checkpointer

    def run(self) -> List[Measurement]:
        num_samples = 0
        start_time = time.perf_counter()
        save_times = Distribution()
        for epoch in range(0, self.epochs):
            for batch_idx, (data, target) in enumerate(self.dataloader):
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(data)
                loss = self.loss_fn(outputs.logits, target)
                loss.backward()
                self.optimizer.step()
                num_samples += len(data)
                if self.checkpoint_criteria(batch_idx):
                    time_to_save = self.checkpointer.checkpoint(self.model, f"batch_{batch_idx}")
                    save_times.add(time_to_save)

        end_time = time.perf_counter()
        training_time = end_time - start_time

        result = [
            Measurement(label="elapsed_time", value=training_time),
            Measurement(label="volume", value=num_samples)
        ]
        if not save_times.is_empty():
            result.append(Measurement(label="save_times", value=save_times))

        return result
