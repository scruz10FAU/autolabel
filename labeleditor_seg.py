import argparse
import json
import tkinter as tk
from PIL import Image, ImageTk
import os

parser = argparse.ArgumentParser(description='YOLO-Seg Label Editor')
parser.add_argument('--image_dir', '-i', type=str, default='training_data/images',
                    help='Directory of training images')
parser.add_argument('--label_dir', '-l', type=str, default='training_data/labels',
                    help='Directory of YOLO-seg label files')
parser.add_argument('--to_update_file', '-u', type=str, default='to_update_zed.txt',
                    help='File listing image paths to review')
parser.add_argument('--classes', '-c', type=str, default='classes.json',
                    help='JSON file mapping class IDs to name and color')
parser.add_argument('--root', '-r', type=str, default=None,
                    help='Root dataset directory containing images/ and labels/ subdirs')
args = parser.parse_args()

if args.root:
    image_dir = os.path.join(args.root, 'images')
    label_dir = os.path.join(args.root, 'labels')
else:
    image_dir = args.image_dir
    label_dir = args.label_dir
to_update_file = args.to_update_file

with open(args.classes) as f:
    raw = json.load(f)
class_names  = {int(k): v['name']  for k, v in raw.items()}
class_colors = {int(k): v['color'] for k, v in raw.items()}
if os.path.exists(to_update_file):
    with open(to_update_file) as f:
        image_files = [os.path.basename(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(image_files)} image(s) from {to_update_file}")
else:
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    print(f"'{to_update_file}' not found — loading all {len(image_files)} images")

index = 0
boxes = []  # each entry: (cls, points_norm, poly_id, text_id)
CLOSE_RADIUS = 8  # px distance to the first vertex that closes a polygon

# in-progress polygon state
current_points = []   # [(x_px, y_px), ...]
current_markers = []  # canvas ids for vertex dots
current_lines = []    # canvas ids for placed edges
rubber_line = None    # canvas id for the line following the cursor

# === GUI SETUP ===
root = tk.Tk()
root.title("YOLO-Seg Label Editor (Mouse Drawing)")

current_class = tk.IntVar(value=0)

canvas_width, canvas_height = 640, 480
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, cursor="tcross")
canvas.pack()

img_tk = None

# === FUNCTIONS ===

def box_color(cls):
    return class_colors.get(int(cls), 'red')

def box_label(cls):
    return class_names.get(int(cls), str(int(cls)))

def draw_polygon_on_canvas(cls, points_norm):
    color = box_color(cls)
    coords = []
    for x, y in points_norm:
        coords.append(x * canvas_width)
        coords.append(y * canvas_height)
    poly_id = canvas.create_polygon(coords, outline=color, fill='', width=2)
    x0, y0 = points_norm[0]
    text_id = canvas.create_text(x0 * canvas_width + 4, y0 * canvas_height - 2,
                                  text=box_label(cls), anchor='sw', fill=color,
                                  font=('Arial', 11, 'bold'))
    return poly_id, text_id

def point_in_polygon(x, y, points_px):
    n = len(points_px)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = points_px[i]
        xj, yj = points_px[j]
        if (yi > y) != (yj > y):
            x_int = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_int:
                inside = not inside
        j = i
    return inside

def cancel_polygon(event=None):
    global current_points, current_markers, current_lines, rubber_line
    for m in current_markers:
        canvas.delete(m)
    for l in current_lines:
        canvas.delete(l)
    if rubber_line is not None:
        canvas.delete(rubber_line)
        rubber_line = None
    current_points = []
    current_markers = []
    current_lines = []

def finish_polygon(event=None):
    global current_points, current_markers, current_lines, rubber_line
    if len(current_points) < 3:
        return
    for m in current_markers:
        canvas.delete(m)
    for l in current_lines:
        canvas.delete(l)
    if rubber_line is not None:
        canvas.delete(rubber_line)
        rubber_line = None

    cls = current_class.get()
    points_norm = [(x / canvas_width, y / canvas_height) for x, y in current_points]
    poly_id, text_id = draw_polygon_on_canvas(cls, points_norm)
    boxes.append((cls, points_norm, poly_id, text_id))

    current_points = []
    current_markers = []
    current_lines = []

def undo_last_point():
    global rubber_line
    if not current_points:
        return
    current_points.pop()
    canvas.delete(current_markers.pop())
    if current_lines:
        canvas.delete(current_lines.pop())
    if rubber_line is not None:
        canvas.delete(rubber_line)
        rubber_line = None

def load_image(idx):
    global img_tk, boxes
    cancel_polygon()
    boxes = []
    canvas.delete("all")

    img_path = os.path.join(image_dir, image_files[idx])
    image = Image.open(img_path).resize((canvas_width, canvas_height))
    img_tk = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    root.title(f"YOLO-Seg Label Editor — {image_files[idx]}  ({idx + 1}/{len(image_files)})")

    name = os.path.splitext(image_files[idx])[0]
    label_path = os.path.join(label_dir, f"{name}.txt")
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7 and len(parts) % 2 == 1:
                    cls = int(float(parts[0]))
                    coords = list(map(float, parts[1:]))
                    points_norm = list(zip(coords[0::2], coords[1::2]))
                    poly_id, text_id = draw_polygon_on_canvas(cls, points_norm)
                    boxes.append((cls, points_norm, poly_id, text_id))

def on_left_click(event):
    x, y = event.x, event.y

    if len(current_points) >= 3:
        sx, sy = current_points[0]
        if (sx - x) ** 2 + (sy - y) ** 2 <= CLOSE_RADIUS ** 2:
            finish_polygon()
            return

    marker = canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill='yellow', outline='black')
    current_markers.append(marker)
    if current_points:
        px, py = current_points[-1]
        current_lines.append(canvas.create_line(px, py, x, y, fill='yellow', width=2))
    current_points.append((x, y))

def on_mouse_move(event):
    global rubber_line
    if not current_points:
        return
    px, py = current_points[-1]
    x, y = event.x, event.y
    if rubber_line is None:
        rubber_line = canvas.create_line(px, py, x, y, fill='yellow', dash=(4, 2), width=1)
    else:
        canvas.coords(rubber_line, px, py, x, y)

def find_box_at(x, y):
    for i, box in reversed(list(enumerate(boxes))):
        points_px = [(px * canvas_width, py * canvas_height) for px, py in box[1]]
        if point_in_polygon(x, y, points_px):
            return i, box
    return None, None

def delete_box_at(x, y):
    i, box = find_box_at(x, y)
    if box is not None:
        canvas.delete(box[2])
        canvas.delete(box[3])
        boxes.pop(i)
        return True
    return False

def on_right_click(event):
    if current_points:
        undo_last_point()
    else:
        delete_box_at(event.x, event.y)

def relabel_box(box_idx, new_cls):
    cls, points_norm, poly_id, text_id = boxes[box_idx]
    color = box_color(new_cls)
    canvas.itemconfig(poly_id, outline=color)
    canvas.itemconfig(text_id, text=box_label(new_cls), fill=color)
    boxes[box_idx] = (new_cls, points_norm, poly_id, text_id)

def on_middle_click(event):
    box_idx, _ = find_box_at(event.x, event.y)
    if box_idx is None:
        return
    menu = tk.Menu(root, tearoff=0)
    for cls_id, cls_name in class_names.items():
        menu.add_command(
            label=cls_name,
            foreground=class_colors[cls_id],
            command=lambda cid=cls_id, bi=box_idx: relabel_box(bi, cid)
        )
    menu.tk_popup(event.x_root, event.y_root)

def save_label():
    name = os.path.splitext(image_files[index])[0]
    label_path = os.path.join(label_dir, f"{name}.txt")
    with open(label_path, 'w') as f:
        for cls, points_norm, _, _ in boxes:
            coords_str = ' '.join(f"{x:.6f} {y:.6f}" for x, y in points_norm)
            f.write(f"{cls} {coords_str}\n")

def next_image():
    global index
    cancel_polygon()
    save_label()
    if index < len(image_files) - 1:
        index += 1
        load_image(index)

def prev_image():
    global index
    cancel_polygon()
    save_label()
    if index > 0:
        index -= 1
        load_image(index)

def clear_boxes():
    global boxes
    cancel_polygon()
    for box in boxes:
        canvas.delete(box[2])
        canvas.delete(box[3])
    boxes = []
    name = os.path.splitext(image_files[index])[0]
    label_path = os.path.join(label_dir, f"{name}.txt")
    if os.path.exists(label_path):
        os.remove(label_path)


# === CLASS SELECTOR ===
class_frame = tk.LabelFrame(root, text="Active Class", padx=6, pady=4)
for cls_id, cls_name in class_names.items():
    color = class_colors[cls_id]
    tk.Radiobutton(
        class_frame, text=cls_name, variable=current_class, value=cls_id,
        fg=color, selectcolor='black', activeforeground=color,
        font=('Arial', 11, 'bold'), indicatoron=True
    ).pack(side=tk.LEFT, padx=8)
class_frame.pack(pady=4)

# === NAV BUTTONS ===
btn_frame = tk.Frame(root)
tk.Button(btn_frame, text="◀ Prev", command=prev_image).pack(side=tk.LEFT, padx=4)
tk.Button(btn_frame, text="Clear Boxes", command=clear_boxes).pack(side=tk.LEFT, padx=4)
tk.Button(btn_frame, text="Next ▶", command=next_image).pack(side=tk.LEFT, padx=4)
btn_frame.pack(pady=5)

canvas.bind("<ButtonPress-1>", on_left_click)
canvas.bind("<Motion>", on_mouse_move)
canvas.bind("<ButtonPress-3>", on_right_click)
canvas.bind("<ButtonPress-2>", on_middle_click)
root.bind("<Return>", finish_polygon)
root.bind("<Escape>", cancel_polygon)

tk.Label(root, text="Left-click: add vertex  |  Click start vertex or Enter: close polygon  |  "
                     "Esc: cancel  |  Right-click: undo vertex / delete  |  Middle-click: relabel",
         font=('Arial', 9), fg='gray').pack(pady=(0, 4))

# === START ===
load_image(index)
root.mainloop()
