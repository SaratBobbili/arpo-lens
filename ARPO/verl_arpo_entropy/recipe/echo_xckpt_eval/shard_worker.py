"""
Per-DP-slot driver for cross-checkpoint ping-pong rollout.

Each shard worker:
  1. Loads a contiguous slice of the val parquet (its shard) into memory.
  2. Spawns two vLLM engine subprocesses (HL + LL), each pinned to its assigned
     physical GPU via CUDA_VISIBLE_DEVICES.
  3. Drives the batched ping-pong loop: every iteration, samples are bucketed
     by their current phase, both buckets are dispatched to their respective
     engine concurrently, returned outputs are folded into per-sample state via
     `pingpong_rollout.apply_turn_output`, LL tool requests are dispatched via a
     thread pool, and the loop continues until every sample is done.
  4. Streams one JSONL row per sample (sample_id, prompt_text, response_text,
     ground_truth, data_source, terminator, turn count) into the shard output
     file passed by the parent driver.

Run as `python -m recipe.echo_xckpt_eval.shard_worker --config <yaml>` where the
config carries combo-specific paths and shard slice; one such subprocess is
spawned per data-parallel slot from `main.py`.
"""

import argparse
import importlib
import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer

from recipe.echo_xckpt_eval.engine_subprocess import READY, engine_target
from recipe.echo_xckpt_eval.pingpong_rollout import (
    ALL_CLOSE_TAGS,
    PHASE_HL,
    PHASE_LL,
    SampleState,
    append_tool_result,
    apply_turn_output,
    hit_tool_cap,
    hit_turn_cap,
)


def _load_tool(class_path: str, params: dict):
    module_path, cls_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(**params)


def _build_prompts(parquet_path, prompt_key, max_prompt_length, system_prompt, tokenizer, shard_lo, shard_hi):
    """Return a list of dicts (sample_id, prompt_text, prompt_ids, ground_truth, data_source) for the shard."""
    df = pd.read_parquet(parquet_path)
    shard = df.iloc[shard_lo:shard_hi].reset_index(drop=True)
    out = []
    for i, row in shard.iterrows():
        msgs = list(row[prompt_key])
        # Mirror RLHFDataset's _replace_system_prompt_row: swap msgs[0]['content']
        # with the active system prompt selected via data.active_system_prompt.
        if system_prompt:
            msgs = [dict(m) for m in msgs]
            msgs[0]["content"] = system_prompt
        prompt_text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(ids) > max_prompt_length:
            # Match rl_dataset.py truncation="error" semantics by left-truncating
            # (val never sees this in practice; valid.parquet prompts are short).
            ids = ids[-max_prompt_length:]
        out.append({
            "sample_id": shard_lo + int(i),
            "prompt_text": prompt_text,
            "prompt_ids": ids,
            "ground_truth": row["reward_model"]["ground_truth"],
            "data_source": row["data_source"],
        })
    return out


def _spawn_engine(ctx, gpu_id, model_path, init_kwargs):
    in_q, out_q = ctx.Queue(), ctx.Queue()
    proc = ctx.Process(target=engine_target, args=(gpu_id, model_path, init_kwargs, in_q, out_q), daemon=False)
    proc.start()
    # Block until the child reports READY so we don't race on the first generate call.
    msg = out_q.get()
    assert msg == READY, f"engine on gpu {gpu_id} reported {msg!r}, expected READY"
    return proc, in_q, out_q


def _generate_concurrent(hl_in, hl_out, ll_in, ll_out, hl_req, ll_req):
    """Send HL+LL requests concurrently; return their outputs (or None if no req)."""
    if hl_req is not None:
        hl_in.put(hl_req)
    if ll_req is not None:
        ll_in.put(ll_req)
    hl_resp = hl_out.get() if hl_req is not None else None
    ll_resp = ll_out.get() if ll_req is not None else None
    return hl_resp, ll_resp


def _build_request(states, max_response_length, sampling_params):
    """Return (prompt_token_ids, sp_dict) or None when there is nothing to send."""
    if not states:
        return None
    prompts = [s.context_ids() for s in states]
    remaining = max((s.remaining_response(max_response_length) for s in states), default=0)
    if remaining <= 0:
        return None
    sp = dict(sampling_params)
    sp["max_tokens"] = remaining
    sp["stop"] = list(ALL_CLOSE_TAGS)
    sp["n"] = 1
    return {"prompt_token_ids": prompts, "sampling_params": sp}


def _run_shard(cfg, samples, tokenizer, hl_path, ll_path, hl_gpu, ll_gpu, output_path):
    # Tools: one shared instance each; thread-safe per their own implementation
    # (BingSearchTool uses fcntl + threading.Lock; PythonTool dispatches subprocesses).
    # Keyed by trigger_tag (e.g. "search", "python") so closing tags map directly.
    tool_instances = {}
    for spec in cfg.xckpt.tools.instances.values():
        tool = _load_tool(spec.class_path, OmegaConf.to_container(spec.params, resolve=True))
        tool_instances[tool.trigger_tag] = tool

    # Subset of engine config that vLLM.LLM accepts; drop layout-only knobs.
    init_kwargs = {
        "tensor_parallel_size": cfg.xckpt.engine.tp,
        "gpu_memory_utilization": cfg.xckpt.engine.gpu_memory_utilization,
        "dtype": cfg.xckpt.engine.dtype,
        "max_model_len": cfg.xckpt.engine.max_model_len,
        "enforce_eager": cfg.xckpt.engine.enforce_eager,
    }

    ctx = mp.get_context("spawn")
    hl_proc, hl_in, hl_out = _spawn_engine(ctx, hl_gpu, hl_path, init_kwargs)
    ll_proc, ll_in, ll_out = _spawn_engine(ctx, ll_gpu, ll_path, init_kwargs)

    sampling_params = OmegaConf.to_container(cfg.xckpt.rollout.val_kwargs, resolve=True)
    sampling_params.pop("n", None)  # n is fixed to 1 per turn; multi-rollout via prompt repetition (val_kwargs.n).

    max_response_length = cfg.xckpt.rollout.max_response_length
    no_progress_terminate = cfg.xckpt.rollout.no_progress_terminate
    max_turns = cfg.xckpt.rollout.max_turns
    tool_call_limit = cfg.xckpt.tools.call_limit
    eos_id = tokenizer.eos_token_id

    # Replicate each sample val_kwargs.n times (interleaved), matching how the
    # trainer's _validate() repeats test_batch before generation.
    n_repeat = int(cfg.xckpt.rollout.val_kwargs.n)
    states: list[SampleState] = []
    sample_meta: list[dict] = []
    for s in samples:
        for r in range(n_repeat):
            states.append(SampleState(sample_id=len(states), prompt_ids=list(s["prompt_ids"])))
            sample_meta.append({**s, "rep": r})

    pool = ThreadPoolExecutor(max_workers=cfg.xckpt.tools.max_workers)
    pbar = tqdm(total=len(states), desc=f"[shard {cfg.xckpt.shard.idx}] samples", position=cfg.xckpt.shard.idx, leave=False)
    completed = 0

    with open(output_path, "w") as fout:
        while True:
            active_idx = [i for i, st in enumerate(states) if st.done is None]
            if not active_idx:
                break

            hl_states = [states[i] for i in active_idx if states[i].phase == PHASE_HL]
            ll_states = [states[i] for i in active_idx if states[i].phase == PHASE_LL]

            hl_req = _build_request(hl_states, max_response_length, sampling_params)
            ll_req = _build_request(ll_states, max_response_length, sampling_params)
            hl_resp, ll_resp = _generate_concurrent(hl_in, hl_out, ll_in, ll_out, hl_req, ll_req)

            tool_jobs = []  # (state, tool_tag, content)

            def _consume(states_phase, resp):
                if resp is None:
                    return
                for st, out in zip(states_phase, resp["outputs"]):
                    outcome = apply_turn_output(
                        st,
                        new_token_ids=out["token_ids"],
                        new_text=out["text"],
                        finish_reason=out["finish_reason"],
                        stop_reason=out["stop_reason"],
                        max_response_length=max_response_length,
                        no_progress_terminate=no_progress_terminate,
                    )
                    if outcome.tool_request is not None:
                        if hit_tool_cap(st, tool_call_limit):
                            # Match training rollout: append EOS + finalize.
                            st.response_ids.append(eos_id)
                            st.result_mask.append(1)
                            st.done = "tool_cap"
                        else:
                            tool_jobs.append((st, *outcome.tool_request))
                    if hit_turn_cap(st, max_turns) and st.done is None:
                        st.done = "turn_cap"

            _consume(hl_states, hl_resp)
            _consume(ll_states, ll_resp)

            # Dispatch tool calls in parallel across all LL-stopped samples.
            if tool_jobs:
                futures = {pool.submit(tool_instances[tag].execute, content): (st, tag) for (st, tag, content) in tool_jobs}
                for fut in futures:
                    st, tag = futures[fut]
                    text = fut.result()
                    if not text:
                        text = f"Tool({tag}) returned empty output."
                    formatted = f" <result>\n{text}\n</result>"
                    result_ids = tokenizer.encode(formatted, add_special_tokens=False)
                    append_tool_result(st, result_ids, max_response_length)

            # Stream-finalize freshly-done samples.
            for i in active_idx:
                st = states[i]
                if st.done is None:
                    continue
                meta = sample_meta[i]
                response_text = tokenizer.decode(st.response_ids, skip_special_tokens=True)
                fout.write(json.dumps({
                    "sample_id": meta["sample_id"],
                    "rep": meta["rep"],
                    "data_source": meta["data_source"],
                    "ground_truth": meta["ground_truth"],
                    "prompt_text": meta["prompt_text"],
                    "response_text": response_text,
                    "terminator": st.done,
                    "turns": st.turn,
                    "tool_calls": st.tool_calls,
                    "select_close_count": st.select_close_count,
                    "response_len": len(st.response_ids),
                }, ensure_ascii=False) + "\n")
                completed += 1
            pbar.n = completed
            pbar.refresh()

    pbar.close()
    hl_in.put(None)
    ll_in.put(None)
    hl_proc.join(timeout=60)
    ll_proc.join(timeout=60)
    pool.shutdown(wait=False, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    tokenizer = AutoTokenizer.from_pretrained(cfg.xckpt.shard.hl_path)
    system_prompt = None
    if cfg.xckpt.data.active_system_prompt is not None:
        sys_yaml = OmegaConf.load(cfg.xckpt.data.system_prompts_yaml)
        system_prompt = sys_yaml.get(f"system_prompt_{cfg.xckpt.data.active_system_prompt}")

    samples = _build_prompts(
        cfg.xckpt.data.val_parquet,
        cfg.xckpt.data.prompt_key,
        cfg.xckpt.data.max_prompt_length,
        system_prompt,
        tokenizer,
        cfg.xckpt.shard.lo,
        cfg.xckpt.shard.hi,
    )

    _run_shard(
        cfg=cfg,
        samples=samples,
        tokenizer=tokenizer,
        hl_path=cfg.xckpt.shard.hl_path,
        ll_path=cfg.xckpt.shard.ll_path,
        hl_gpu=cfg.xckpt.shard.hl_gpu,
        ll_gpu=cfg.xckpt.shard.ll_gpu,
        output_path=cfg.xckpt.shard.output_path,
    )


if __name__ == "__main__":
    main()
