import hydra
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../conf/scenarios", config_name="dataloading")
def run_experiment(config: DictConfig):
    scenario = infer_scenario()
    print(OmegaConf.to_yaml(config))


def infer_scenario() -> str:
    config = HydraConfig.get()
    return config.job.config_name



if __name__ == '__main__':
    run_experiment()
