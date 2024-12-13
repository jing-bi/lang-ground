import gradio as gr
from langground import LangGround
import torch
def create_model(loc_model, llm_model):
    return LangGround(loc_model=loc_model, llm_model=llm_model)


# Default model
model = create_model(loc_model="yolo", llm_model="Qwen/Qwen2.5-7B-Instruct")


title = """
<center> 

<h1>  🔍 Language Localization </h1>
<b> Upload an image and ask questions to find objects in it. <b>

</center>
"""
with gr.Blocks() as demo:
    gr.HTML(title)

    with gr.Row():
        with gr.Column(scale=1):
            loc_model_input = gr.Dropdown(
                choices=["yolo", "owl"],
                value="yolo",
                label="Localization Model",
            )
            llm_model_input = gr.Dropdown(
                choices=["Qwen/Qwen2.5-7B-Instruct", "OpenGVLab/InternVL2_5-8B", 
                         "OpenGVLab/InternVL2_5-4B", "OpenGVLab/InternVL2_5-2B",
                         "OpenGVLab/InternVL2_5-1B"],
                value="OpenGVLab/InternVL2_5-2B",
                label="LLM Model",
            )
        with gr.Column(scale=1):
            frame_input = gr.Image(type="pil", label="Upload Frame")

        with gr.Column(scale=1):
            threshold_input = gr.Slider(minimum=0, maximum=1, value=0.4, step=0.1, label="Threshold")
            question_input = gr.Textbox(lines=2, placeholder="Enter your question here", label="Question")
            objs = gr.Textbox(label="Answer")
            submit_btn = gr.Button("Submit")

    with gr.Row():
        all_bbox_image = gr.Image(label="Found Objects")
        llm_bbox_image = gr.Image(label="Selected Objects")

    def update_model_and_localize(frame, question, threshold, loc_model, llm_model):
        global model
        # 删除旧模型并清理显存
        if model is not None:
            del model
            torch.cuda.empty_cache()
        # 创建新模型
        model = create_model(loc_model, llm_model)
        return model.localize(frame, question, threshold=threshold)


    submit_btn.click(
        fn=update_model_and_localize,
        inputs=[frame_input, question_input, threshold_input, loc_model_input, llm_model_input],
        outputs=[objs, all_bbox_image, llm_bbox_image],
    )
    examples = gr.Examples(
        examples=[
            ["assets/demo.jpeg", "I'm thirsty"],
            ["assets/kitchen.webp", "The food has expired and is no longer safe to eat."],
            ["assets/kitchen.webp", "The food is about to expire."],
        ],
        inputs=[frame_input, question_input],
    )
demo.queue().launch()
