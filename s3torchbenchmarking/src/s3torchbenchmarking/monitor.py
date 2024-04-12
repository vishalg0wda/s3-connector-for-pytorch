import threading
import time
from abc import ABCMeta
from collections import defaultdict
from dataclasses import dataclass
from typing import Union, Dict, Any, List

import psutil

from .common import Distribution


@dataclass(frozen=True)
class Measurement:
    label: str
    value: Union[float, Distribution]

    def encode(self) -> Dict[str, Any]:
        v = self.value
        if isinstance(v, Distribution):
            v_dist: Distribution = v
            v = {"stats": v_dist.summarize()}

        return {
            self.label: v
        }

    def __repr__(self):
        return "Measurement({0})".format(self.encode())


class Sensor(metaclass=ABCMeta):
    def __init__(self, label: str):
        self.label = label

    def instrument(self) -> float:
        raise NotImplementedError()


class Monitor:
    def __init__(self, sensors: List[Sensor], sleep_time_s: float = 0.05):
        self.sensors = sensors
        self._measurements = defaultdict(lambda: Distribution())
        self._monitor_thread = None
        self._stop_event = threading.Event()
        self.sleep_time_s = sleep_time_s

    def _monitor(self):
        for sensor in self.sensors:
            self._measurements[sensor].add(sensor.instrument())

    def run(self):
        while not self._stop_event.is_set():
            self._monitor()
            time.sleep(self.sleep_time_s)

    def start(self):
        self._monitor_thread = threading.Thread(target=self.run)
        self._monitor_thread.start()

    def stop(self):
        self._stop_event.set()
        self._monitor_thread.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def measurements(self) -> List[Measurement]:
        return [
            Measurement(label=sensor.label, value=distro) for sensor, distro in self._measurements.items()
        ]


class CPUUtilSensor(Sensor):

    def __init__(self):
        super().__init__(label="cpu_util")

    def instrument(self) -> float:
        return psutil.cpu_percent()


class CPUMemorySensor(Sensor):

    def __init__(self):
        super().__init__(label="cpu_memory")

    def instrument(self) -> float:
        return psutil.virtual_memory().percent
