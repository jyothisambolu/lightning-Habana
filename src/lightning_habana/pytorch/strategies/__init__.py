# Copyright The Lightning AI team.
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

from lightning_habana.pytorch.strategies.ddp import HPUDDPStrategy
from lightning_habana.pytorch.strategies.deepspeed import HPUDeepSpeedStrategy
from lightning_habana.pytorch.strategies.fsdp import HPUFSDPStrategy
from lightning_habana.pytorch.strategies.parallel import HPUParallelStrategy
from lightning_habana.pytorch.strategies.single import SingleHPUStrategy

__all__ = ["HPUFSDPStrategy", "HPUDDPStrategy", "HPUDeepSpeedStrategy", "HPUParallelStrategy", "SingleHPUStrategy"]
