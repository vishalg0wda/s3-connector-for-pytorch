import json
from typing import List, Optional, Callable

from .monitor import Measurement, Monitor
from .scenarios import Scenario

MeasurementDeriver = Callable[[List[Measurement]], Measurement]


class BenchmarkResult:

    def __init__(self, measurements: List[Measurement], derivers=None):
        if derivers is None:
            derivers = [derive_throughput]
        self.measurements = measurements
        derived_measurements = [deriver(measurements) for deriver in derivers]
        # Filter out empty measurements(when dependent fields were not present)
        derived_measurements = filter(lambda e: e, derived_measurements)
        self.measurements += derived_measurements

    def to_json(self) -> str:
        fields = {}
        for measurement in self.measurements:
            fields.update(measurement.encode())

        return json.dumps(fields)


class BenchmarkRunner:
    def __init__(self, monitor: Monitor):
        self._monitor = monitor

    def run(self, scenario: Scenario, measurement_derivers: List[MeasurementDeriver] = None) -> BenchmarkResult:
        with self._monitor:
            measurements = scenario.run()
        measurements += self._monitor.measurements

        return BenchmarkResult(measurements,
                               # A common derived measurement is `throughput`
                               derivers=measurement_derivers)


def derive_throughput(measurements: List[Measurement]) -> Optional[Measurement]:
    volume_measurement: Optional[Measurement] = next(
        filter(lambda measurement: measurement.label == 'volume', measurements), None)
    elapsed_time_measurement: Optional[Measurement] = next(
        filter(lambda measurement: measurement.label == 'elapsed_time', measurements), None)
    if volume_measurement and elapsed_time_measurement:
        return Measurement(label="throughput", value=volume_measurement.value / elapsed_time_measurement.value)
