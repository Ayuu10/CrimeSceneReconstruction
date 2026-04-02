import spacy
import re

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_visual_elements(text: str) -> str:
    """
    Stage 0: Visual Scene Extraction.
    Removes subjective emotions, psychological states, and legal jargon.
    Leaves physical objects, attributes, and spatial layouts.
    """
    # Simple dictionary to replace legal/forensic jargon with generic visual nouns
    replacements = {
        r"\bsuspect\b": "person", 
        r"\bSuspect\b": "Person",
        r"\bvictim\b": "person",
        r"\bVictim\b": "Person",
        r"\bperpetrator\b": "person",
        r"\bwitness\b": "person"
    }
    
    for pattern, rep in replacements.items():
        text = re.sub(pattern, rep, text)
        
    # Vocabulary to filter out
    abstract_lemmas = {"panic", "flee", "motive", "statement", "say", "believe", "think", "feel", "opinion"}
    emotion_adjs = {"angry", "scared", "fearful", "sad", "happy", "panicked", "terrified"}
    
    doc = nlp(text)
    
    cleaned_tokens = []
    for token in doc:
        if token.lemma_.lower() in abstract_lemmas:
            continue
        if token.pos_ == "ADJ" and token.lemma_.lower() in emotion_adjs:
            continue
            
        cleaned_tokens.append(token.text_with_ws)
        
    cleaned_text = "".join(cleaned_tokens).strip()
    
    # Normalize extra whitespaces caused by removal
    cleaned_text = re.sub(r' +', ' ', cleaned_text)
    
    return cleaned_text

def process_text(cleaned_text: str):
    """
    Stage 1: NLP Preprocessing
    Returns the spaCy Doc object containing tokens, POS tags, and dependency tree.
    """
    doc = nlp(cleaned_text)
    return doc

def run_nlp_pipeline(narrative: str):
    """
    Combines Stage 0 and 1. Returns the cleaned text and spaCy Doc.
    """
    cleaned_text = extract_visual_elements(narrative)
    doc = process_text(cleaned_text)
    return cleaned_text, doc

if __name__ == "__main__":
    sample = "The suspect fled the scene in a panic. A bloody knife was left on top of the wooden table. A broken chair was lying next to the table. The witness felt very scared."
    clean, doc = run_nlp_pipeline(sample)
    print("Cleaned Narrative:", clean)
    for token in doc:
        print(f"{token.text} ({token.pos_}, {token.dep_}, {token.head.text})")
