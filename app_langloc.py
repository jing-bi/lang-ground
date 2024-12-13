import gradio as gr
from langground import LangGround

model = LangGround()

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
            frame_input = gr.Image(type="pil", label="Upload Frame")

        with gr.Column(scale=1):
            threshold_input = gr.Slider(minimum=0, maximum=1, value=0.4, step=0.1, label="Threshold")
            question_input = gr.Textbox(lines=2, placeholder="Enter your question here", label="Question")
            objs = gr.Textbox(label="Answer")

    with gr.Row():
        all_bbox_image = gr.Image(label="Found Objects")
        llm_bbox_image = gr.Image(label="Selected Objects")

    submit_btn = gr.Button("Submit")
    submit_btn.click(
        fn=lambda f, q, t: model.localize(f, q, threshold=t),
        inputs=[frame_input, question_input, threshold_input],
        outputs=[objs, all_bbox_image, llm_bbox_image],
    )
    examples = gr.Examples(examples=[["assets/demo.jpeg", "I'm thirsty"]], inputs=[frame_input, question_input])
demo.queue().launch()
