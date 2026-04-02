import gradio as gr
import sys
import os
import networkx as nx

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from nlp_preprocessing import run_nlp_pipeline
from scene_graph import build_scene_graph
from layout_generation import generate_layout
from diffusion_generation import load_pipeline, generate_image

# Load model globally
print("Loading Stable Diffusion and ControlNet pipeline...")
pipe = load_pipeline()
print("Model loaded successfully.")

def process_narrative(text):
    clean_prompt, doc = run_nlp_pipeline(text)
    
    G = build_scene_graph(doc)
    graph_str = ""
    for u, v, data in G.edges(data=True):
        graph_str += f"[{u}] --({data['relation']})--> [{v}]\n"
        
    if not graph_str and G.nodes:
        graph_str = "Isolated objects: " + ", ".join(G.nodes)
        
    if not graph_str:
        graph_str = "No physical objects extracted."
        
    layout_img = generate_layout(G)
    
    refined_prompt = "crime scene photo, highly detailed, realistic, photorealistic, 4k, " + clean_prompt
    neg_prompt = "cartoon, illustration, 3d render, text, watermark, low quality, abstract"
    
    result_img = generate_image(pipe, refined_prompt, negative_prompt=neg_prompt)
    
    return clean_prompt, graph_str, layout_img, result_img

with gr.Blocks(title="Crime Scene Generator") as demo:
    gr.Markdown("# Crime Scene Image Generator from Police Narratives")
    gr.Markdown("Enter a police report narrative. The system will extract visual elements, build a scene graph, generate a spatial bounding box layout, and finally use ControlNet to generate a realistic crime scene image.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(lines=5, label="Police Narrative Report", placeholder="e.g. The suspect fled the scene in a panic. A bloody knife was left on top of the wooden table. A broken chair was lying next to the table.")
            generate_btn = gr.Button("Generate Scene")
            
            clean_out = gr.Textbox(label="Stage 1: Extracted Visual Description", interactive=False)
            graph_out = gr.Textbox(lines=5, label="Stage 2: Extracted Scene Graph Relations", interactive=False)
            
        with gr.Column(scale=1):
            layout_out = gr.Image(label="Stage 3: Generated ControlNet Layout Map", type="pil")
            result_out = gr.Image(label="Stage 5: Generated Crime Scene Result", type="pil")
            
    generate_btn.click(
        fn=process_narrative,
        inputs=input_text,
        outputs=[clean_out, graph_out, layout_out, result_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
