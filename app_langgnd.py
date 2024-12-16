from functools import partial
from pathlib import Path
import uuid
import cv2
import gradio as gr
from langground import LangGround, get_gr_video_current_time, text_palette


state = {"loc_model": None, "llm_model": None, "model": None}
style = """
    #gr_title {text-align: center;}
    #gr_video {max-height: 100vh;}
    #gr_chatbot {max-height: 50vh;}
"""


title = """
<center> 

<h1>  👌 Language Grounding </h1>
<b> Upload a video and ask questions about objects in the video. </b>

</center>
"""


class Worker:
    def __init__(self):
        self.grounder = LangGround(llm_model="gpt-40-mini", track=True)
        self.reset()

    def reset(self):
        self.questions = [None]
        self.stamp = 0
        self.status = "init"
        self.video_chunk_name = None
        self.video_chunk_writer = None
        self.latest_frame = None
        self.frame_idx = 0

    def log(self, action, timestamp):
        if action in ["pause", "play"]:
            self.status = action
        self.stamp = timestamp

    def process_frame(self, frame):
        return self.grounder.track(frame)

    def response(self, message):
        gr.Info(f"Start with question: {message}")
        self.questions.append(message)
        objs, _ = self.grounder.track(self.latest_frame, message, threshold=0.25)
        return objs

    def create_output(self, codec, fps, width, height):
        path = Path(__file__).parent / "cache"
        self.video_chunk_name = str(path / f"{uuid.uuid4()}.mp4")
        self.video_chunk_writer = cv2.VideoWriter(self.video_chunk_name, codec, fps, (width, height))

    def yield_buffer(self, buffer):
        if buffer:
            for frm in buffer:
                self.video_chunk_writer.write(frm)
            buffer.clear()
            self.video_chunk_writer.release()
        result_chunk = self.video_chunk_name
        self.create_output(self.codec, self.fps, self.w, self.h)
        return result_chunk

    def process_video(self, video):
        cap = cv2.VideoCapture(video)
        self.codec = cv2.VideoWriter_fourcc(*"mp4v")
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        buffer = []
        self.create_output(self.codec, self.fps, self.w, self.h)
        self.frame_idx = 0
        while True:
            while self.frame_idx - 1 >= self.stamp * self.fps:
                if buffer and self.status == "pause":
                    yield self.yield_buffer(buffer)

            ret, frame = cap.read()
            self.latest_frame = frame
            if not ret:
                break
            self.frame_idx += 1
            processed = self.process_frame(self.latest_frame)
            if self.frame_idx % self.fps == 0:
                gr.Info(f"frame {self.frame_idx} processed")
            buffer.append(processed)
            if self.status == "play" and len(buffer) > self.fps:
                yield self.yield_buffer(buffer)
        cap.release()


worker = Worker()


with gr.Blocks(css=style) as demo:
    gr.HTML(title)
    with gr.Row():
        with gr.Column():
            gr_video = gr.Video(label="video stream", elem_id="gr_video", visible=True, sources=["upload"], autoplay=False)
        with gr.Column():
            gr_processed_video = gr.Video(
                label="Processed Video", elem_id="gr_processed_video", visible=True, interactive=False, streaming=True, autoplay=True
            )
        with gr.Column():
            gr_text = gr.Textbox(label="Question", placeholder="Ask about objects in the video...")
            objs = gr.Highlightedtext(show_legend=False, show_inline_category=False, color_map=text_palette, label="Objects Found")
            gr_submit = gr.Button("Submit")
            gr_submit.click(fn=worker.response, inputs=[gr_text], outputs=[objs])

        gr_video_time = gr.Number(value=0, visible=False)
        gr_video.change(worker.process_video, inputs=[gr_video], outputs=[gr_processed_video])
        gr.Timer(value=1).tick(partial(worker.log, "auto"), inputs=[gr_video_time], js=get_gr_video_current_time)
        gr_video.play(partial(worker.log, "play"), inputs=[gr_video_time], js=get_gr_video_current_time)
        gr_video.pause(partial(worker.log, "pause"), inputs=[gr_video_time], js=get_gr_video_current_time)
    gr.Examples(examples=[["assets/tools.mp4"]], inputs=[gr_video])
if __name__ == "__main__":
    demo.launch()
