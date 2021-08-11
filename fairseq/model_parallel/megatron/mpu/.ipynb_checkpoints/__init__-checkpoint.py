# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Model parallel utility interface."""

from fairseq.model_parallel.megatron.mpu.cross_entropy import vocab_parallel_cross_entropy

from fairseq.model_parallel.megatron.mpu.data import broadcast_data

from fairseq.model_parallel.megatron.mpu.grads import clip_grad_norm

from fairseq.model_parallel.megatron.mpu.initialize import is_unitialized
from fairseq.model_parallel.megatron.mpu.initialize import destroy_model_parallel
from fairseq.model_parallel.megatron.mpu.initialize import get_data_parallel_group
from fairseq.model_parallel.megatron.mpu.initialize import get_data_parallel_rank
from fairseq.model_parallel.megatron.mpu.initialize import get_data_parallel_world_size
from fairseq.model_parallel.megatron.mpu.initialize import get_model_parallel_group
from fairseq.model_parallel.megatron.mpu.initialize import get_model_parallel_rank, set_model_parallel_rank
from fairseq.model_parallel.megatron.mpu.initialize import get_model_parallel_src_rank
from fairseq.model_parallel.megatron.mpu.initialize import get_model_parallel_world_size, set_model_parallel_world_size
from fairseq.model_parallel.megatron.mpu.initialize import initialize_model_parallel
from fairseq.model_parallel.megatron.mpu.initialize import model_parallel_is_initialized

from fairseq.model_parallel.megatron.mpu.layers import LayerNorm
from fairseq.model_parallel.megatron.mpu.layers import ColumnParallelLinear
from fairseq.model_parallel.megatron.mpu.layers import RowParallelLinear
from fairseq.model_parallel.megatron.mpu.layers import VocabParallelEmbedding

from fairseq.model_parallel.megatron.mpu.mappings import copy_to_model_parallel_region
from fairseq.model_parallel.megatron.mpu.mappings import gather_from_model_parallel_region
from fairseq.model_parallel.megatron.mpu.mappings import reduce_from_model_parallel_region
from fairseq.model_parallel.megatron.mpu.mappings import scatter_to_model_parallel_region

from fairseq.model_parallel.megatron.mpu.random import checkpoint
from fairseq.model_parallel.megatron.mpu.random import get_cuda_rng_tracker
from fairseq.model_parallel.megatron.mpu.random import init_checkpointed_activations_memory_buffer
from fairseq.model_parallel.megatron.mpu.random import model_parallel_cuda_manual_seed
from fairseq.model_parallel.megatron.mpu.random import reset_checkpointed_activations_memory_buffer

from fairseq.model_parallel.megatron.mpu.utils import divide
from fairseq.model_parallel.megatron.mpu.utils import split_tensor_along_last_dim
