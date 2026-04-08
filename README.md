# Crime Scene Image Generator from Police Narratives

This project is an advanced AI pipeline that converts narrative police reports and crime scene descriptions into realistic, highly detailed crime scene images. It leverages a multi-stage approach, seamlessly blending Natural Language Processing (NLP), semantic scene graph construction, spatial heuristics, and Stable Diffusion XL.

## 🚀 Features & Pipeline Stages

1. **Stage 0 & 1: NLP Preprocessing**  
   Uses `spaCy` to parse the narrative. It algorithmically filters out subjective elements ("panic", "fear") and normalizes forensic/legal jargon ("suspect", "victim", "witness") into visually neutral equivalents ("person").

2. **Stage 2: Semantic Scene Graph Construction** 
   Extracts physical entities (supporting compound nouns like "coffee cup") and deduplicates coreferences (e.g., merging "table" and "wooden table"). Using `spacy` dependency parsing, it extracts spatial relationships (e.g., "on top of", "next to", "under") linked by prepositions or verbs, and stores them in a `networkx` DiGraph. Background environments ("sidewalk", "parking lot") are dynamically categorized.

3. **Stage 3: Spatial Layout Generation**  
   Iterates through the `networkx` scene graph to render a 2D bounding-box layout using `PIL`. The module evaluates spatial connections to intuitively position items (e.g. objects related by "on" spawn within the parent's boundaries). It leverages an aggressive 6-angle collision detection mechanism to ensure isolated objects never overlap improperly.

4. **Stage 4 & 5: Stable Diffusion XL Generation**  
   Utilizes the `diffusers` library in combination with Stable Diffusion XL 1.0 (`stabilityai/stable-diffusion-xl-base-1.0`). To bypass the standard 77-token CLIP limit of regular Stable Diffusion models, this pipeline uses the **`compel`** library. This allows for infinitely long, highly robust police narratives without dropping context at the end of the paragraph.

5. **Interface**  
   A beautiful, fast web UI built entirely in `Gradio` allowing you to step through and visualize the Output Prompt, the Scene Graph Edges, the Bounding Box Layout Map, and the final Generated Crime Scene Image.

## 🛠️ Technology Stack
- **NLP**: `spaCy` (en_core_web_sm)
- **Data Structures**: `NetworkX`
- **Computer Vision**: `Pillow` (PIL)
- **Generative AI**: `Diffusers`, `PyTorch`, `Compel`, Stable Diffusion XL 1.0
- **Web UI**: `Gradio`

## 💻 Installation

### Running Locally (Requires High-End GPU)
```bash
# Clone the repository
git clone https://github.com/Ayuu10/CrimeSceneReconstruction.git
cd CrimeSceneReconstruction

# Create and activate a Virtual Environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Download the spaCy language model
python -m spacy download en_core_web_sm

# Launch the Application
python app.py
```
*Note: Depending on your internet speed, the first run will download the ~6GB SDXL Base model.*

### Running on Google Colab (Free GPU)
To easily run this project at lightning speeds using a free NVIDIA T4 GPU:
1. Open a new [Google Colab Notebook](https://colab.research.google.com/).
2. Change the Runtime Type to **T4 GPU**.
3. Run the following code block to clone, setup, and run:

```python
!git clone https://github.com/Ayuu10/CrimeSceneReconstruction.git
%cd CrimeSceneReconstruction
!pip install -r requirements.txt
!python -m spacy download en_core_web_sm
!sed -i 's/demo.launch(server_name="0.0.0.0", server_port=7860)/demo.launch(share=True)/g' app.py
!python app.py
```
Colab will output a public link (e.g. `https://xxxxxx.gradio.live`)! Click it to open the generator.

## 🎯 Example Prompt
*“A man is standing near a wooden table on the sidewalk outside a café. A knife is on the table next to a coffee cup. A chair is beside the table, and a backpack is under the chair. A street lamp is above the table lighting the area. A parked motorcycle is near the edge of the sidewalk behind the table.”*
