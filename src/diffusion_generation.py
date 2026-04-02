import torch
from diffusers import StableDiffusionXLPipeline, UniPCMultistepScheduler
from PIL import Image
from compel import Compel, ReturnedEmbeddingsType

def load_pipeline():
    """
    Initializes the standard Stable Diffusion pipeline (without ControlNet).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load SDXL Pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" # Helps load SDXL faster to VRAM
    )
    
    # Speed up inference
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    if device == "cuda":
        # Enable memory savings
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
        
    return pipe

def generate_image(pipe, prompt: str, negative_prompt: str = "") -> Image.Image:
    """
    Stage 4 & 5: Standard Diffusion Generation
    """
    # Setup compel for SDXL bypassing 77 token limit
    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2], 
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2], 
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, 
        requires_pooled=[False, True]
    )
    
    # Parse infinitely long prompts
    conditioning, pooled = compel(prompt)
    negative_conditioning, negative_pooled = compel(negative_prompt)
    
    # Ensure they have the same shape if chunk counts mismatch
    [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
    
    # Generate image
    generator = torch.manual_seed(42)
    output = pipe(
        prompt_embeds=conditioning,
        pooled_prompt_embeds=pooled,
        negative_prompt_embeds=negative_conditioning,
        negative_pooled_prompt_embeds=negative_pooled,
        num_inference_steps=50,
        generator=generator
    ).images[0]
    
    return output

if __name__ == "__main__":
    from nlp_preprocessing import run_nlp_pipeline
    
    text = "A bloody knife was left on top of the wooden table. A broken chair was lying next to the table."
    # Stage 0 & 1
    clean_prompt, doc = run_nlp_pipeline(text)
    
    # Stage 4 & 5
    print("Loading pipeline...")
    pipe = load_pipeline()
    
    refined_prompt = "crime scene photo, highly detailed, realistic, " + clean_prompt
    neg_prompt = "cartoon, illustration, 3d render, text, watermark, low quality, abstract"
    
    print("Generating image...")
    result_img = generate_image(pipe, refined_prompt, negative_prompt=neg_prompt)
    result_img.save("sample_result.png")
    print("Saved sample_result.png")
