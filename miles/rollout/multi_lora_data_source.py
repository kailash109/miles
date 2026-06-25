"""Multi-LoRA data source: round-robins ``get_samples`` across ACTIVE
adapters only. Reaching ``num_epoch`` triggers ``deregister_adapter``;
the trainer's lifecycle gate drives the rest of the state machine.
"""

import copy
import logging
from argparse import Namespace

import ray

from miles.ray.multi_lora_controller import get_multi_lora_controller
from miles.rollout.data_source import DataSource, RolloutDataSource
from miles.utils.adapter_config import AdapterConfig, AdapterState
from miles.utils.types import AdapterRef, RewardSpec, Sample

logger = logging.getLogger(__name__)


class MultiLoRADataSource(DataSource):
    def __init__(self, args: Namespace):
        self.args = args
        self.controller = get_multi_lora_controller()
        self.sources: dict[str, RolloutDataSource] = {}
        self.configs: dict[str, AdapterConfig] = {}
        self.epoch_counts: dict[str, int] = {}
        self._reconcile(self._fetch_configs())

    def _fetch_configs(self) -> dict[str, AdapterConfig]:
        return ray.get(self.controller.adapter_configs.remote())

    def _reconcile(self, configs: dict[str, AdapterConfig]) -> None:
        for name in list(self.sources):
            if name not in configs:
                del self.sources[name]
                del self.configs[name]
                del self.epoch_counts[name]
                logger.info(f"Removed data source for adapter '{name}'")

        for name, config in configs.items():
            if name not in self.sources:
                self.sources[name] = self._create_adapter_source(config)
                self.epoch_counts[name] = 0
                logger.info(f"Created data source for adapter '{name}' from {config.data}")
            self.configs[name] = config

    def _create_adapter_source(self, config: AdapterConfig) -> RolloutDataSource:
        adapter_args = copy.copy(self.args)
        adapter_args.prompt_data = config.data
        adapter_args.input_key = config.input_key or self.args.input_key
        adapter_args.label_key = config.label_key or self.args.label_key
        adapter_args.metadata_key = config.metadata_key or self.args.metadata_key
        if config.seed is not None:
            adapter_args.rollout_seed = config.seed
        return RolloutDataSource(adapter_args)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        configs = self._fetch_configs()
        self._reconcile(configs)

        active_names = [n for n in self.sources if configs[n].state == AdapterState.ACTIVE]
        datasource_drained = [n for n in self.sources if configs[n].state == AdapterState.DRAINING_DATASOURCE]

        assert len(active_names) + len(datasource_drained) > 0, "get_samples called without any active adapters"

        # Run one last iter for those being drained, since sglang rollout needs to be able to run one last
        # time for the adapter after the draining has kicked off in order to update the adapter states
        active_names += datasource_drained
        per_adapter = num_samples // len(active_names)
        remainder = num_samples % len(active_names)

        refs = {name: AdapterRef(name=name, slot=configs[name].slot) for name in active_names}
        reward_specs = {
            name: RewardSpec(rm_type=configs[name].rm_type, custom_rm_path=configs[name].custom_rm_path)
            for name in active_names
        }

        # Get samples from each data source
        all_samples: list[list[Sample]] = []
        for i, name in enumerate(active_names):
            # TODO: remove early source bias from this
            count = per_adapter + (1 if i < remainder else 0)
            if count == 0:
                continue

            source: RolloutDataSource = self.sources[name]
            config = configs[name]
            prev_epoch = source.epoch_id

            adapter_samples = source.get_samples(count)

            # Begin deregistration process when out of data
            # sample_group_index is the same as tracking the row index
            # Default to length of dataset, override if num rollout is set
            default_num_row = (getattr(config, "num_epoch", 1) or 1) * len(source.dataset)
            num_row = getattr(config, "num_row") or default_num_row
            if source.sample_group_index >= num_row:
                logger.info(f"Adapter '{name}' reached num_row={num_row}, deregistering")
                print(f"Adapter '{name}' reached num_row={num_row}, deregistering...")
                datasource_drained.append(name)

            # Add LoRA adapter data + per adapter reward fn data
            ref = refs[name]
            reward_spec = reward_specs[name]
            for group in adapter_samples:
                for sample in group:
                    sample.adapter = ref
                    sample.reward_spec = reward_spec
            all_samples.extend(adapter_samples)

        if datasource_drained:
            ray.get(self.controller.update_adapter_state.remote(datasource_drained, AdapterState.DRAINING_INFLIGHT))

        return all_samples

    def add_samples(self, samples: list[list[Sample]]):
        """Re-queue retried groups; drops groups for non-ACTIVE adapters."""
        configs = self._fetch_configs()
        self._reconcile(configs)

        for group in samples:
            name = group[0].adapter.name if group and group[0].adapter else None
            if not name or name not in self.sources:
                continue
            config = self.configs.get(name)
            if config is None or config.state != AdapterState.ACTIVE:
                continue
            self.sources[name].add_samples([group])

    def save(self, rollout_id):
        for source in self.sources.values():
            source.save(rollout_id)

    def load(self, rollout_id=None):
        for source in self.sources.values():
            source.load(rollout_id)
