import contextlib
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh

aten = torch.ops.aten


def sdpa_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    # extract local tensor and sharding infos to a OpInfo
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    # sharding propagation
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"

    local_results = _scaled_dot_product_flash_attention(
        op_call,
        op_info.mesh,
        *op_info.local_args,
        **op_info.local_kwargs,
    )

    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


def _scaled_dot_product_flash_attention(
    op_call: torch._ops.OpOverload,
    mesh: DeviceMesh,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, ...]:
    if return_debug_mask:
        raise NotImplementedError("return_debug_mask is not supported yet")
    if is_causal:
        raise NotImplementedError("is_causal is not supported yet")

    pg = mesh.get_group()
    assert isinstance(pg, dist.ProcessGroup), "must be single dimension"
    rank = dist.get_rank(pg)
    size = dist.get_world_size(pg)

    # rank 0 sends to rank 1, rank 1 sends to rank 2, ..., rank n-1 sends to rank 0
    right_dsts = list(range(1, size)) + [0]

    next_kv = None

    chunks = []
    logsumexps = []
    for i in range(size):
        # overlap communication with compute
        if next_kv is not None:
            next_kv = ft_c.wait_tensor(next_kv)
            key = next_kv[: key.numel()].reshape(key.shape)
            value = next_kv[key.numel() :].reshape(value.shape)

        if i < (size - 1):
            next_kv = torch.cat([key.flatten(), value.flatten()])
            next_kv = ft_c.permute_tensor(next_kv, right_dsts, pg)

        local_results = op_call(
            query,
            key,
            value,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )
        chunks.append(local_results[0])
        logsumexps.append(local_results[1])

    softmax_lse = torch.empty_like(logsumexps[0])
    for lse in logsumexps:
        softmax_lse += lse.exp()
    softmax_lse = softmax_lse.log_()

    out = torch.zeros_like(chunks[0])
    for chunk, chunk_lse in zip(chunks, logsumexps):
        softmax_lse_corrected = torch.exp(chunk_lse - softmax_lse)
        out_corrected = chunk * softmax_lse_corrected.unsqueeze(-1)
        out += out_corrected

    local_results = (out, softmax_lse) + local_results[2:]
    return local_results


customized_ops = {
    aten._scaled_dot_product_flash_attention.default: sdpa_handler,
}


@contextlib.contextmanager
def attention_parallel() -> None:
    """
    This enables attention parallel optimizations. Currently only ring attention for SDPA flash attention.
    """
    DTensor._op_dispatcher._custom_op_handlers.update(customized_ops)

    yield

    for custom_op in customized_ops:
        DTensor._op_dispatcher._custom_op_handlers.pop(custom_op)
