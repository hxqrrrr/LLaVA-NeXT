from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from transformers import BitsAndBytesConfig
import warnings
from PIL import Image
import gradio as gr
import torch
import copy
import os

warnings.filterwarnings("ignore", category=FutureWarning)

# 定义4位量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model_path = "/root/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-7b/snapshots/4481d270cc22fd5c4d1bb5df129622006ccd9234"
# 尝试使用本地模型路径，如果不存在则从huggingface下载
if os.path.exists(model_path):
    model_path = model_path
else:
    model_path = "liuhaotian/llava-v1.5-7b"
    
# 修改加载模型的参数
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path,
    None,
    model_name="llava-v1.5",
    quantization_config=quantization_config,
    use_safetensors=True,
    local_files_only=True,
    device_map="auto"
)

def process_image(image, question):
    if image is None:
        return "请先上传图片"
    
    # 处理图片 - 修改设备分配方式
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [img.to(dtype=torch.float16, device=model.device) for img in image_tensor]
    
    # 准备提示词
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = copy.deepcopy(conv_templates["llava_v1.5"])
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # 生成回答
    input_ids = tokenizer_image_token(
        prompt, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors="pt"
    ).unsqueeze(0).to(model.device)
    
    output = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        do_sample=True,
        temperature=0.2,
        max_new_tokens=128,  # 减少token数以节省显存
    )
    
    response = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return response

# 创建Gradio界面
demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="上传图片"),
        gr.Textbox(label="请输入你的问题", placeholder="请描述这张图片...")
    ],
    outputs=gr.Textbox(label="模型回答"),
    title="LLaVA Demo",
    description="上传一张图片，然后输入问题，模型会给出回答。",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)