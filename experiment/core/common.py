from abc import ABC
import hydra
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
from typing import Optional

__all__ = ['FileIO', 'Serialization']

class Serialization(ABC):
    @classmethod
    def from_config_dict(cls, config: DictConfig):
        """Instantiates object using DictConfig-based configuration"""
        # Resolve the config dict
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
            config = OmegaConf.create(config)
            OmegaConf.set_struct(config, True)


        if ('cls' in config or 'target' in config) and 'params' in config:
            # regular hydra-based instantiation
            instance = hydra.utils.instantiate(config=config)
        elif '_target_' in config:
            # regular hydra-based instantiation
            instance = hydra.utils.instantiate(config=config)
        else:
            # models are handled differently for now
            instance = cls(cfg=config)

        if not hasattr(instance, '_cfg'):
            instance._cfg = config
        return instance

    def to_config_dict(self) -> DictConfig:
        """Returns object's configuration to config dictionary"""
        if hasattr(self, '_cfg') and self._cfg is not None and isinstance(self._cfg, DictConfig):
            # Resolve the config dict
            config = OmegaConf.to_container(self._cfg, resolve=True)
            config = OmegaConf.create(config)
            OmegaConf.set_struct(config, True)

            self._cfg = config

            return self._cfg
        else:
            raise NotImplementedError(
                'to_config_dict() can currently only return object._cfg but current object does not have it.'
            )


class FileIO(ABC):
    def save_to(self, save_path: str):
        """Saves module/model with weights"""
        raise NotImplementedError()

    @classmethod
    def restore_from(
        cls,
        restore_path: str,
        override_config_path: Optional[str] = None,
        map_location: Optional['torch.device'] = None,
        strict: bool = True,
    ):
        """Restores module/model with weights"""
        raise NotImplementedError()

    @classmethod
    def from_config_file(cls, path2yaml_file: str):
        """
        Instantiates an instance of NeMo Model from YAML config file.
        Weights will be initialized randomly.
        Args:
            path2yaml_file: path to yaml file with model configuration
        Returns:
        """
        if issubclass(cls, Serialization):
            conf = OmegaConf.load(path2yaml_file)
            return cls.from_config_dict(config=conf)
        else:
            raise NotImplementedError()

    def to_config_file(self, path2yaml_file: str):
        """
        Saves current instance's configuration to YAML config file. Weights will not be saved.
        Args:
            path2yaml_file: path2yaml_file: path to yaml file where model model configuration will be saved
        Returns:
        """
        if hasattr(self, '_cfg'):
            # self._cfg = maybe_update_config_version(self._cfg)
            with open(path2yaml_file, 'w') as fout:
                OmegaConf.save(config=self._cfg, f=fout, resolve=True)
        else:
            raise NotImplementedError()
