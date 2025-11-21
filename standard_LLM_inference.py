from vllm import LLM, SamplingParams, EngineArgs
from transformers import AutoTokenizer
from vllm.utils.argparse_utils import FlexibleArgumentParser



def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="Qwen/Qwen3-1.7b")
    parser.set_defaults(gpu_memory_utilization=0.7)
    parser.set_defaults(max_model_len=8192)
    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)  
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)

    return parser

def main(args:dict):
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7b", trust_remote_code=True)


    llm = LLM(**args)
    # llm = LLM(
    #     model=model_id,
    #     trust_remote_code=True,
    #     gpu_memory_utilization=0.8, 
    #     max_model_len=8192
    # )
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k





    print("\n" + "="*20 + " 场景 A: Without Chat Template (续写) " + "="*20)

    # vLLM 的优势：可以直接传一个 List，自动做 Batch 推理
    raw_prompts = [
        "In a distant future, artificial intelligence became",
        "The quick brown fox jumps over",
    ]
    


    outputs = llm.generate(raw_prompts, sampling_params)

    for output in outputs:
        print(f"Input: {output.prompt!r}")
        print(f"Output: {output.outputs[0].text!r}")
        print("-" * 30)


    # ==========================================
    # 场景 B: With Chat Template (对话 / Instruction)
    # 适用于：问答、聊天
    # ==========================================
    print("\n" + "="*20 + " 场景 B: With Chat Template (对话) " + "="*20)

    # 1. 定义对话列表
    conversations = [
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "Tell me a joke about Python."}],
        
        [{"role": "user", "content": "What is the capital of France?"}]
    ]

    # 2. 使用 apply_chat_template (关键修改！)
    # 区别：在 vLLM 中，我们设置 tokenize=False
    # 目的：我们需要把对话变成“字符串”，然后喂给 vLLM，让 vLLM 自己去处理 Token
    chat_prompts = [
        tokenizer.apply_chat_template(
            convo, 
            tokenize=False,             
            add_generation_prompt=True  
        ) 
        for convo in conversations
    ]

    print(f"[Debug] Converted Prompt Example:\n{chat_prompts[0]!r}\n")


    outputs = llm.generate(chat_prompts, sampling_params)

    for output in outputs:
        print(f"Generated Response: {output.outputs[0].text}")
        print("-" * 30)

if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)