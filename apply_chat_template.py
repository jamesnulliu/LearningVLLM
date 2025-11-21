import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 设置模型 ID (可以是本地路径，也可以是 HuggingFace Hub ID)
model_id = "Qwen/Qwen3-1.7b"

# 2. 加载 Tokenizer 和 模型
# trust_remote_code=True 对于某些自定义架构模型是必须的
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    dtype=torch.bfloat16, # 推荐使用 bfloat16 或 float16 以节省显存
    device_map="auto"           # 自动分配 GPU
)

# 3. 定义对话历史 (Standard Chat Format)
# 这是人类可读的通用格式，不包含任何特定模型的特殊符号
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "解释一下为什么天空是蓝色的？请用简短的一句话回答。"},
]

# 4. 关键步骤：使用 apply_chat_template
# 这一步将 List[Dict] 转换为模型能听懂的 Tensor 输入
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,              # 直接转换为数字 ID
    add_generation_prompt=True, # 关键：添加 assistant 引导符，告诉模型"该你说话了"
    return_tensors="pt"         # 返回 PyTorch Tensor
).to(model.device)

# --- (可选) 调试：看看模版到底把你的对话变成了什么样 ---
# print("DEBUG - 实际输入给模型的 Prompt:")
print(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))


# 5. 模型推理
print("Generating...")
outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

# 6. 解码输出
# response 包含了输入+输出，我们需要把输入部分切掉，只看新生成的部分
generated_ids = outputs[0][len(input_ids[0]):]
response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print("-" * 50)
print(f"User Input: {messages[1]['content']}")
print(f"AI Response: {response_text}")
print("-" * 50)