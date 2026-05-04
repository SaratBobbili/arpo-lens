"""
vLLM engine subprocess target.

Each instance owns one vLLM `LLM` engine pinned to a single GPU via
`CUDA_VISIBLE_DEVICES`. The shard worker spawns two of these per data-parallel
slot (one for HL, one for LL) so the in-process layout is:

  shard_worker (parent, no GPU)
    ├── engine subprocess (HL on physical GPU `gpu_id`, sees only cuda:0 inside)
    └── engine subprocess (LL on physical GPU `gpu_id`, sees only cuda:0 inside)

Communication via two `multiprocessing.Queue`s. Requests are dicts
`{"prompt_token_ids": list[list[int]], "sampling_params": dict}`; responses are
`{"outputs": list[dict]}` where each per-sample dict carries `token_ids`,
`finish_reason`, `stop_reason`, and decoded `text`. A `None` request signals
shutdown.
"""

import os


# Sentinel sent by the child to confirm the engine is loaded and ready to serve.
READY = "__engine_ready__"


def engine_target(
    gpu_id: int,
    model_path: str,
    init_kwargs: dict,
    in_queue,
    out_queue,
):
    # CRITICAL: must be set before importing torch / vllm so the underlying
    # CUDA driver only ever sees the assigned device. With this set, the engine
    # always loads on cuda:0 inside this process (the physical GPU we asked for).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tensor_parallel_size=init_kwargs.get("tensor_parallel_size", 1),
        gpu_memory_utilization=init_kwargs.get("gpu_memory_utilization", 0.7),
        dtype=init_kwargs.get("dtype", "bfloat16"),
        max_model_len=init_kwargs.get("max_model_len", 5760),
        enforce_eager=init_kwargs.get("enforce_eager", False),
        # Disabled: prevents vLLM from spawning helper processes that would try
        # to grab additional GPUs through Ray.
        distributed_executor_backend=None,
        # Don't disable prefix caching: ping-pong shares a long common prefix
        # across turns, so prefix caching is exactly the win.
        enable_prefix_caching=True,
        disable_log_stats=True,
        trust_remote_code=init_kwargs.get("trust_remote_code", False),
        seed=init_kwargs.get("seed", 0),
    )
    out_queue.put(READY)

    while True:
        req = in_queue.get()
        if req is None:
            break

        sp_kwargs = dict(req["sampling_params"])
        # `include_stop_str_in_output=True` keeps the matched close tag in both
        # the decoded text and the token_ids, which the ping-pong logic relies
        # on (stop tag is the last thing emitted; we strip back to it).
        sp_kwargs.setdefault("include_stop_str_in_output", True)
        sampling_params = SamplingParams(**sp_kwargs)

        outputs = llm.generate(
            prompt_token_ids=req["prompt_token_ids"],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        # vLLM 0.8 returns RequestOutputs in the same order as the input prompts.
        results = [
            {
                "token_ids": list(o.outputs[0].token_ids),
                "text": o.outputs[0].text,
                "finish_reason": o.outputs[0].finish_reason,
                "stop_reason": o.outputs[0].stop_reason,
            }
            for o in outputs
        ]
        out_queue.put({"outputs": results})
