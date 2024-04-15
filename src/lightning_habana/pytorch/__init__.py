# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from lightning_habana.pytorch.accelerator import HPUAccelerator
from lightning_habana.pytorch.datamodule.datamodule import HPUDataModule
from lightning_habana.pytorch.plugins.deepspeed_precision import HPUDeepSpeedPrecisionPlugin
from lightning_habana.pytorch.plugins.fsdp_precision import HPUFSDPPrecision
from lightning_habana.pytorch.plugins.io_plugin import HPUCheckpointIO
from lightning_habana.pytorch.plugins.precision import HPUPrecisionPlugin
from lightning_habana.pytorch.profiler.profiler import HPUProfiler
from lightning_habana.pytorch.strategies.ddp import HPUDDPStrategy
from lightning_habana.pytorch.strategies.deepspeed import HPUDeepSpeedStrategy
from lightning_habana.pytorch.strategies.parallel import HPUParallelStrategy
from lightning_habana.pytorch.strategies.single import SingleHPUStrategy
from lightning_habana.pytorch.strategies.fsdp import HPUFSDPStrategy

__all__ = [
    "HPUAccelerator",
    "HPUDDPStrategy",
    "HPUDeepSpeedStrategy",
    "HPUParallelStrategy",
    "SingleHPUStrategy",
    "HPUFSDPStrategy",
    "HPUPrecisionPlugin",
    "HPUCheckpointIO",
    "HPUProfiler",
    "HPUDataModule",
    "HPUDeepSpeedPrecisionPlugin",
    "HPUFSDPPrecision"
]
