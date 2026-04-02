import sys
import os

# Ensure src is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nlp_preprocessing import run_nlp_pipeline
from scene_graph import build_scene_graph
from layout_generation import generate_layout
from diffusion_generation import load_pipeline, generate_image

def generate_crime_scene(narrative: str, pipe):
    print("Stage 0 & 1: NLP Preprocessing...")
    clean_prompt, doc = run_nlp_pipeline(narrative)
    print(f"Cleaned prompt: {clean_prompt}")
    
    print("Stage 2: Scene Graph Construction...")
    G = build_scene_graph(doc)
    print(f"Extracted {len(G.nodes)} objects and {len(G.edges)} relationships.")
    
    print("Stage 3: Spatial Layout Generation...")
    layout_img = generate_layout(G)
    
    print("Stage 4 & 5: Image Diffusion Generation...")
    refined_prompt = "crime scene photo, highly detailed, realistic, " + clean_prompt
    neg_prompt = "cartoon, illustration, 3d render, text, watermark, low quality, abstract"
    
    result_img = generate_image(pipe, refined_prompt, negative_prompt=neg_prompt)
    
    return clean_prompt, G, layout_img, result_img

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate crime scene from narrative.")
    parser.add_argument("--text", type=str, required=True, help="Police narrative text")
    parser.add_argument("--output", type=str, default="output.png", help="Output file path")
    
    args = parser.parse_args()
    
    print("Loading AI Models (this may take a minute)...")
    pipe = load_pipeline()
    
    _, _, layout, result = generate_crime_scene(args.text, pipe)
    
    layout.save("layout_" + args.output)
    result.save(args.output)
    
    print(f"Successfully generated layout_{args.output} and {args.output}")
