from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import gradio as gr
import torch
import copy

# 使用公开可用的模型
model_path = "lmms-lab/llama3-llava-next-8b"  # 改用这个模型
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path,
    None,
    model_name="llava_llama3",  # 注意这里也要相应修改
    device_map="auto"
)

def process_image(image, question):
    if image is None:
        return "请先上传图片"
    
    # 处理图片
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [img.to(dtype=torch.float16, device="cuda") for img in image_tensor]
    
    # 准备提示词
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = copy.deepcopy(conv_templates["llava_llama_3"])  # 使用对应的模板
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # 生成回答
    input_ids = tokenizer_image_token(
        prompt, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors="pt"
    ).unsqueeze(0).cuda()
    
    output = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        do_sample=True,
        temperature=0.2,
        max_new_tokens=512,
    )
    
    response = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return response

# 创建 Gradio 界面
demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="上传图片"),
        gr.Textbox(label="请输入你的问题", placeholder="请描述这张图片...")
    ],
    outputs=gr.Textbox(label="模型回答"),
    title="LLaVA-NeXT Demo",
    description="上传一张图片，然后输入问题，模型会给出回答。",
)

# 启动服务
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)