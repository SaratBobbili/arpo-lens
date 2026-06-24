import sys
import os
sys.path.append(os.getcwd())
import asyncio
import argparse
import nltk
nltk.download('punkt')

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Asynchronous inference engine")
    
    vllm_group = parser.add_argument_group("VLLM Configuration")
    vllm_group.add_argument("--endpoints", type=str, nargs='+', required=True,
                            help="List of VLLM endpoints")
    vllm_group.add_argument("--model_path", type=str, required=True,
                            help="Model path for tokenizer loading")
    vllm_group.add_argument("--api_keys", type=str, nargs='+', default=None,
                            help="List of API keys corresponding to endpoints")
    vllm_group.add_argument("--default_model", type=str, default="AgentModel",
                            help="Default model name to use")
    
    generation_group = parser.add_argument_group("Generation Parameters")
    generation_group.add_argument("--temperature", type=float, default=0.6,
                                  help="Temperature for generation")
    generation_group.add_argument("--max_tokens", type=int, default=4096,
                                  help="Maximum number of new tokens to generate")
    generation_group.add_argument("--top_p", type=float, default=0.95,
                                  help="Top-p sampling cutoff")
    generation_group.add_argument("--top_k", type=int, default=20,
                                  help="Top-k sampling cutoff")
    generation_group.add_argument("--min_p", type=float, default=0.0,
                                  help="Minimum probability threshold")
    generation_group.add_argument("--repetition_penalty", type=float, default=1.1,
                                  help="Repetition penalty factor")
    generation_group.add_argument("--include_stop_str_in_output", type=bool, default=True,
                                  help="Whether to include stop strings in output")

    inference_group = parser.add_argument_group("Inference Configuration")
    inference_group.add_argument("--max_concurrent_requests", type=int, default=32,
                                 help="Maximum number of concurrent samples to process")
    inference_group.add_argument("--dataset_name", type=str, required=True, nargs='+',
                                 help="List of dataset names (separated by space)")
    inference_group.add_argument("--output_path", type=str, required=True,
                                 help="Root directory for saving results. Dataset results are saved at /root/dataset_name/dataset_name_output_i.json")
    inference_group.add_argument("--prompt_type", type=str, default='echo',
                                 help="Prompt type (echo, code_search, search, math, base)")
    inference_group.add_argument("--counts", type=int, default=500,
                                 help="Number of samples to process")
    inference_group.add_argument("--data_path", type=str, default="data",
                                 help="Custom data path. Datasets are expected at /root_path/dataset_name/test.jsonl")
    inference_group.add_argument("--max_tool_calls", type=int, default=6,
                                 help="Maximum number of tool invocations")
    inference_group.add_argument("--sample_timeout", type=int, default=300,
                                 help="Timeout in seconds for processing a single sample")

    tools_group = parser.add_argument_group("Tool Configuration")
    tools_group.add_argument("--conda_path", type=str, required=True,
                             help="Path to Conda installation")
    tools_group.add_argument("--conda_env", type=str, required=True,
                             help="Conda environment name")
    return parser.parse_args()

def get_inference_instance():
    args = parse_arguments()
    print(vars(args))
    from src.inference_engine import AsyncInference as AsyncInfer
    inference = AsyncInfer(args)
    return inference

async def main():
    inference = get_inference_instance()
    await inference.run()
    sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
