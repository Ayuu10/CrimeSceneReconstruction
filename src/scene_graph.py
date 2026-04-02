import networkx as nx
from spacy.tokens import Doc, Token

# Words that shouldn't be boxed
IGNORE_LEMMAS = {
    "crime", "scene", "night", "day", "area", "parking", "convenience", 
    "street", "ground", "store", "wall", "light", "incident", "location", 
    "place", "time", "front", "back", "side", "top", "bottom", "edge",
    "corner", "center", "middle", "distance", "vicinity", "storefront",
    "building", "room", "floor", "ceiling", "window", "door", "darkness",
    "view", "sight", "shadow", "shape", "form", "surface", "picture", "photo",
    "sidewalk", "cafe", "café", "restaurant", "hospital", "school", "park", "road"
}

def get_entity_name(token: Token) -> str:
    """Extracts the noun and its adjectival/compound modifiers."""
    modifiers = []
    for w in token.children:
        if w.dep_ in ('amod', 'compound'):
            modifiers.append(w.lemma_.lower())
    name = " ".join(modifiers + [token.lemma_.lower()])
    return name

def is_valid_object(token: Token) -> bool:
    if token.pos_ not in ("NOUN", "PROPN"):
        return False
    if token.lemma_.lower() in IGNORE_LEMMAS:
        return False
    return True

def build_scene_graph(doc: Doc) -> nx.DiGraph:
    G = nx.DiGraph()
    
    # Track all valid noun chunks that we care about
    nouns = [tok for tok in doc if is_valid_object(tok)]
    
    # Coreference / Deduplication dict
    resolved_names = {}
    for noun in nouns:
        raw_name = get_entity_name(noun)
        best_match = raw_name
        for existing in set(resolved_names.values()):
            # If the head noun matches exactly, treat as same object (e.g. "table" == "wooden table")
            # Prefer the longer, more descriptive name
            if noun.lemma_.lower() in existing.split() or existing.split()[-1] == noun.lemma_.lower():
                best_match = existing if len(existing) > len(raw_name) else raw_name
                
                # Update existing mappings to the longer name if we found a better/longer name
                for k, v in list(resolved_names.items()):
                    if v == existing:
                        resolved_names[k] = best_match
                break
                
        resolved_names[noun] = best_match
        
    for noun, resolved in resolved_names.items():
        G.add_node(resolved, label=resolved)
        
    for token in doc:
        # Case 1: Preposition attached to a noun
        if is_valid_object(token):
            subj_name = resolved_names.get(token)
            if not subj_name: continue
            for child in token.children:
                if child.dep_ == "prep":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj" and is_valid_object(pobj):
                            obj_name = resolved_names.get(pobj)
                            if obj_name:
                                relation = child.lower_
                                G.add_edge(subj_name, obj_name, relation=relation)
                            
        # Case 2: Preposition attached to a verb
        if token.pos_ in ("VERB", "AUX"):
            # find subjects
            subjects = [c for c in token.children if "subj" in c.dep_]
            preps = [c for c in token.children if c.dep_ == "prep"]
            
            for subj in subjects:
                if not is_valid_object(subj):
                    continue
                subj_name = resolved_names.get(subj)
                if not subj_name: continue
                
                for prep in preps:
                    relation = prep.lower_
                    advmods = [c.lower_ for c in token.children if c.dep_ in ("advmod", "amod", "prt")]
                    if advmods:
                        relation = " ".join(advmods) + " " + relation
                        
                    for pobj in prep.children:
                        if pobj.dep_ == "pobj":
                            if is_valid_object(pobj):
                                obj_name = resolved_names.get(pobj)
                                if obj_name:
                                    G.add_edge(subj_name, obj_name, relation=relation)
                            elif pobj.lemma_.lower() in ("top", "front", "back", "side", "middle", "center", "edge"):
                                # Look deeper
                                deeper_preps = [c for c in pobj.children if c.dep_ == "prep"]
                                if deeper_preps:
                                    dp = deeper_preps[0]
                                    relation = relation + " " + pobj.lemma_.lower() + " " + dp.lower_
                                    for dpobj in dp.children:
                                        if dpobj.dep_ == "pobj" and is_valid_object(dpobj):
                                            obj_name = resolved_names.get(dpobj)
                                            if obj_name:
                                                G.add_edge(subj_name, obj_name, relation=relation)
                            
    return G
