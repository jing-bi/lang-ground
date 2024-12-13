import gradio as gr
from langground import LangGround, text_palette


state = {"loc_model": None, "llm_model": None, "model": None}


def load_model(loc_model: str, llm_model: str) -> LangGround:
    if (loc_model, llm_model) != (state["loc_model"], state["llm_model"]):
        gr.Info("Loading models...", duration=5)
        state.update({"model": LangGround(loc_model=loc_model, llm_model=llm_model), "loc_model": loc_model, "llm_model": llm_model})
        gr.Info("Models loaded!", duration=2.5)
    return state["model"]


def predict(frame, question: str, threshold: float, loc_model: str, llm_model: str):
    if not frame or not question.strip():
        gr.Warning("Please provide both an image and a question")
        return "", None, None

    model = load_model(loc_model, llm_model)
    return model.localize(frame, question, threshold=threshold)


title = """
<center> 

<h1>  🔍 Language Localization </h1>
<b> Upload an image and ask questions to find objects in it. <b>

</center>
"""

css = """.my-group {max-width: 600px !important; max-height: 600px !important;}
         .my-column {display: flex !important; justify-content: center !important; align-items: center !important;}"""

with gr.Blocks(css=css) as demo:
    gr.HTML(title)

    with gr.Row():

        with gr.Column(scale=1):
            frame_input = gr.Image(type="pil", label="Upload Frame")

        with gr.Column(scale=1):
            with gr.Row():
                with gr.Column(scale=1):

                    loc_model_input = gr.Dropdown(
                        choices=["yolo", "owl"],
                        value="yolo",
                        label="Localization Model",
                    )
                with gr.Column(scale=2):

                    llm_model_input = gr.Dropdown(
                        choices=[
                            "Qwen/Qwen2.5-7B-Instruct",
                            "OpenGVLab/InternVL2_5-8B",
                            "OpenGVLab/InternVL2_5-4B",
                            "OpenGVLab/InternVL2_5-2B",
                            "OpenGVLab/InternVL2_5-1B",
                        ],
                        value="Qwen/Qwen2.5-7B-Instruct",
                        label="LLM Model",
                    )
            threshold_input = gr.Slider(minimum=0, maximum=1, value=0.4, step=0.1, label="Threshold")
            question_input = gr.Textbox(lines=2, placeholder="Enter your question here", label="Question")
            objs = gr.Highlightedtext(show_legend=False, show_inline_category=False, color_map=text_palette, label="Objects Found")
            submit_btn = gr.Button("Submit")

    with gr.Row():
        all_bbox_image = gr.Image(label="Found Objects")
        llm_bbox_image = gr.Image(label="Selected Objects")

    submit_btn.click(
        fn=predict,
        inputs=[frame_input, question_input, threshold_input, loc_model_input, llm_model_input],
        outputs=[objs, all_bbox_image, llm_bbox_image],
    )
    with gr.Row():
        examples = gr.Examples(
            examples=[
                ["assets/demo.jpeg", "I'm thirsty"],
                ["assets/kitchen.webp", "The food has expired and is no longer safe to eat."],
                ["assets/kitchen.webp", "The food is about to expire."],
            ],
            inputs=[frame_input, question_input],
        )
if __name__ == "__main__":
    demo.launch()
