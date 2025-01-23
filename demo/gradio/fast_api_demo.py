from fastapi import FastAPI
import gradio as gr

app = FastAPI()

def greet(name):
    return f"Hello, {name}!"
    
gradio_interface = gr.Interface(fn=greet, inputs="text", outputs="text")
gradio_app = gr.routes.App.create_app(gradio_interface)

app.mount("/gradio", gradio_app)
