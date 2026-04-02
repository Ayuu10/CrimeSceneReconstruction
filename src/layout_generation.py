import networkx as nx
from PIL import Image, ImageDraw, ImageFont
import random

def get_default_size(name: str):
    name = name.lower()
    if any(w in name for w in ["table", "bed", "sofa", "couch", "car"]):
        return (250, 150)
    elif any(w in name for w in ["chair", "person", "man", "woman", "body", "officer"]):
        return (100, 200)
    elif any(w in name for w in ["bicycle", "bike", "motorcycle"]):
        return (180, 120)
    elif any(w in name for w in ["lamp", "street lamp", "light"]):
        return (40, 200)
    elif any(w in name for w in ["backpack", "bag", "box"]):
        return (70, 90)
    elif any(w in name for w in ["cup", "coffee", "knife", "gun", "glass", "bottle"]):
        return (40, 40)
    else:
        return (60, 60)

def check_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return not (x1 + w1 + 5 <= x2 or x2 + w2 + 5 <= x1 or y1 + h1 + 5 <= y2 or y2 + h2 + 5 <= y1)

def find_non_overlapping_position(cw, ch, canvas_size, placed_boxes, max_attempts=100):
    for _ in range(max_attempts):
        cx = random.randint(10, max(11, canvas_size[0] - cw - 10))
        cy = random.randint(10, max(11, canvas_size[1] - ch - 10))
        new_box = (cx, cy, cw, ch)
        if not any(check_overlap(new_box, box) for box in placed_boxes):
            return cx, cy
    return random.randint(10, max(11, canvas_size[0] - cw - 10)), random.randint(10, max(11, canvas_size[1] - ch - 10))

def generate_layout(graph: nx.DiGraph, canvas_size=(512, 512)) -> Image.Image:
    canvas = Image.new("RGB", canvas_size, "black")
    draw = ImageDraw.Draw(canvas)
    boxes = {}
    
    nodes_by_degree = sorted(graph.degree, key=lambda x: x[1], reverse=True)
    if not nodes_by_degree:
        return canvas
        
    for node, _ in nodes_by_degree:
        w, h = get_default_size(node)
        placed_parent = None
        relation = ""
        
        for u in graph.predecessors(node):
            if u in boxes:
                placed_parent = u
                relation = graph.edges[u, node].get("relation", "").lower()
                break
                
        if not placed_parent:
            for v in graph.successors(node):
                if v in boxes:
                    placed_parent = v
                    relation = graph.edges[node, v].get("relation", "").lower()
                    break
                    
        if placed_parent:
            px, py, pw, ph = boxes[placed_parent]
            if "on" in relation or "top" in relation:
                # Place inside the parent box to represent "on table" or "on floor"
                cx = px + (pw - w) // 2 + random.randint(-15, 15)
                cy = py + (ph - h) // 2 + random.randint(-40, 0)
            elif "under" in relation or "below" in relation:
                cx = px + (pw - w) // 2
                cy = py + ph - (h // 3)
            elif "above" in relation or "over" in relation:
                cx = px + (pw - w) // 2
                cy = py - h - 10
            elif "next" in relation or "beside" in relation:
                cx = px + pw + 10
                cy = py + ph - h
            else:
                cx, cy = find_non_overlapping_position(w, h, canvas_size, list(boxes.values()))
        else:
            if not boxes:
                cx = (canvas_size[0] - w) // 2
                cy = (canvas_size[1] - h) // 2
            else:
                cx, cy = find_non_overlapping_position(w, h, canvas_size, list(boxes.values()))
                
        cx = max(0, min(canvas_size[0] - w, cx))
        cy = max(0, min(canvas_size[1] - h, cy))
        boxes[node] = (cx, cy, w, h)
              
    try: font = ImageFont.truetype("arial.ttf", 16)
    except IOError: font = ImageFont.load_default()
        
    for node, (x, y, w, h) in boxes.items():
        draw.rectangle([x, y, x+w, y+h], outline="white", width=3)
        draw.text((x + 5, y + 5), graph.nodes[node].get("label", node), fill="white", font=font)
        
    return canvas
