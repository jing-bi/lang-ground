


import gradio as gr
from langground import LangGround

model = LangGround()

def gradio_interface(frame, question):
    objs, all_bbox_image, llm_bbox_image = model.localize(frame, question)
    return objs, all_bbox_image, llm_bbox_image

frame_input = gr.Image(type="pil", label="Upload Frame")
question_input = gr.Textbox(lines=2, placeholder="Enter your question here", label="Question")
objs = gr.Textbox(label="Answer")
all_bbox_image = gr.Image(label="all_bbox_image")
llm_bbox_image = gr.Image(label="llm_bbox_image")
examples = [
    ["assets/demo.jpeg", "I'm thirsty"]
]
gr.Interface(fn=gradio_interface, 
             inputs=[frame_input, question_input], 
             outputs=[objs, all_bbox_image, llm_bbox_image],
             title="Visual Question Answering",
             description="Upload a frame and ask a question about the objects in the frame.",
             examples=examples).launch()


