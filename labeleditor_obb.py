import argparse
import json
import math
import tkinter as tk
from PIL import Image, ImageTk
import os

parser = argparse.ArgumentParser(description='YOLO-OBB Label Editor')
parser.add_argument('--image_dir', '-i', type=str, default='training_data/images',
                    help='Directory of training images')
parser.add_argument('--label_dir', '-l', type=str, default='training_data/labels',
                    help='Directory of YOLO-OBB label files')
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
boxes = []  # each entry: (cls, points_norm, poly_id, text_id) — points_norm is 4 corners

# in-progress oriented-box state (built from 3 clicks: edge start, edge end, width)
obb_points = []        # [(x_px, y_px), ...] — up to 2 entries (edge start, edge end)
obb_point_markers = [] # canvas ids for the two corner dots
obb_edge_line = None   # canvas id for the fixed first-edge line
rubber_line = None     # canvas id for the line following the cursor (while placing edge end)
preview_shape = None   # canvas id for the live rectangle preview (while placing width)

# existing-box selection + rotate-handle state
selected_idx = None    # index into boxes of the currently selected rectangle, or None
dragging_handle = False
handle_id = None       # canvas id for the draggable rotate handle
handle_line_id = None  # canvas id for the line from box center to the handle
HANDLE_HIT_RADIUS = 10  # px, how close a click must be to grab the handle
HANDLE_GAP = 20          # px, how far the handle sits beyond the box's corner radius

# === GUI SETUP ===
root = tk.Tk()
root.title("YOLO-OBB Label Editor (Mouse Drawing)")

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

def compute_obb_corners(a, b, p):
    """Rectangle with edge a->b as one side, extruded toward p."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length == 0:
        return None
    dirx, diry = dx / length, dy / length
    perpx, perpy = -diry, dirx
    px, py = p[0] - ax, p[1] - ay
    width = px * perpx + py * perpy
    c = (bx + perpx * width, by + perpy * width)
    d = (ax + perpx * width, ay + perpy * width)
    return [a, b, c, d]

def cancel_obb(event=None):
    global obb_points, obb_point_markers, obb_edge_line, rubber_line, preview_shape
    for m in obb_point_markers:
        canvas.delete(m)
    obb_point_markers = []
    if obb_edge_line is not None:
        canvas.delete(obb_edge_line)
        obb_edge_line = None
    if preview_shape is not None:
        canvas.delete(preview_shape)
        preview_shape = None
    if rubber_line is not None:
        canvas.delete(rubber_line)
        rubber_line = None
    obb_points = []

def undo_obb_point():
    global obb_points, obb_edge_line, rubber_line, preview_shape
    if not obb_points:
        return
    obb_points.pop()
    if obb_point_markers:
        canvas.delete(obb_point_markers.pop())
    if obb_edge_line is not None:
        canvas.delete(obb_edge_line)
        obb_edge_line = None
    if preview_shape is not None:
        canvas.delete(preview_shape)
        preview_shape = None
    if rubber_line is not None:
        canvas.delete(rubber_line)
        rubber_line = None

def finish_obb(p):
    global obb_points, obb_point_markers, obb_edge_line, rubber_line, preview_shape
    a, b = obb_points
    corners = compute_obb_corners(a, b, p)
    if corners is None:
        return
    for m in obb_point_markers:
        canvas.delete(m)
    obb_point_markers = []
    if obb_edge_line is not None:
        canvas.delete(obb_edge_line)
        obb_edge_line = None
    if preview_shape is not None:
        canvas.delete(preview_shape)
        preview_shape = None
    if rubber_line is not None:
        canvas.delete(rubber_line)
        rubber_line = None

    cls = current_class.get()
    points_norm = [(x / canvas_width, y / canvas_height) for x, y in corners]
    poly_id, text_id = draw_polygon_on_canvas(cls, points_norm)
    boxes.append((cls, points_norm, poly_id, text_id))
    obb_points = []

def rect_metrics(points_norm):
    """Decompose 4 stored corners into center, unit edge/perp axes, and half-extents (px space)."""
    a, b, c, _d = [(x * canvas_width, y * canvas_height) for x, y in points_norm]
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    edge_dir = (dx / length, dy / length) if length else (1.0, 0.0)
    perp_dir = (-edge_dir[1], edge_dir[0])
    hw = length / 2
    wx, wy = c[0] - b[0], c[1] - b[1]
    width = wx * perp_dir[0] + wy * perp_dir[1]
    hh = width / 2
    mid_ab = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    center = (mid_ab[0] + perp_dir[0] * hh, mid_ab[1] + perp_dir[1] * hh)
    return center, edge_dir, perp_dir, hw, hh

def corners_from_metrics(center, edge_dir, perp_dir, hw, hh):
    cx, cy = center
    ex, ey = edge_dir
    px, py = perp_dir
    a = (cx - ex * hw - px * hh, cy - ey * hw - py * hh)
    b = (cx + ex * hw - px * hh, cy + ey * hw - py * hh)
    c = (cx + ex * hw + px * hh, cy + ey * hw + py * hh)
    d = (cx - ex * hw + px * hh, cy - ey * hw + py * hh)
    return [a, b, c, d]

def handle_position(points_norm):
    center, _edge_dir, perp_dir, hw, hh = rect_metrics(points_norm)
    r = math.hypot(hw, hh) + HANDLE_GAP
    return center, (center[0] - perp_dir[0] * r, center[1] - perp_dir[1] * r)

def clear_selection_handle():
    global handle_id, handle_line_id
    if handle_id is not None:
        canvas.delete(handle_id)
        handle_id = None
    if handle_line_id is not None:
        canvas.delete(handle_line_id)
        handle_line_id = None

def draw_selection_handle(idx):
    global handle_id, handle_line_id
    clear_selection_handle()
    _cls, points_norm, _poly_id, _text_id = boxes[idx]
    center, handle_pos = handle_position(points_norm)
    handle_line_id = canvas.create_line(center[0], center[1], handle_pos[0], handle_pos[1],
                                         fill='cyan', dash=(3, 2))
    hx, hy = handle_pos
    handle_id = canvas.create_oval(hx - 6, hy - 6, hx + 6, hy + 6, fill='cyan', outline='black')

def select_box(idx):
    global selected_idx
    selected_idx = idx
    draw_selection_handle(idx)

def deselect(event=None):
    global selected_idx
    selected_idx = None
    clear_selection_handle()

def handle_hit(x, y):
    if selected_idx is None or handle_id is None:
        return False
    hx1, hy1, hx2, hy2 = canvas.coords(handle_id)
    hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
    return (hcx - x) ** 2 + (hcy - y) ** 2 <= HANDLE_HIT_RADIUS ** 2

def rotate_selected_to(mouse_x, mouse_y):
    idx = selected_idx
    cls, points_norm, poly_id, text_id = boxes[idx]
    center, _edge_dir, _perp_dir, hw, hh = rect_metrics(points_norm)
    dxm, dym = mouse_x - center[0], mouse_y - center[1]
    dist = math.hypot(dxm, dym)
    if dist < 1e-6:
        return
    # handle direction points away from the box; the perpendicular axis is its opposite
    perp_new = (-dxm / dist, -dym / dist)
    edge_new = (perp_new[1], -perp_new[0])
    corners_px = corners_from_metrics(center, edge_new, perp_new, hw, hh)
    points_norm_new = [(x / canvas_width, y / canvas_height) for x, y in corners_px]

    coords = []
    for x, y in corners_px:
        coords.extend([x, y])
    canvas.coords(poly_id, *coords)
    x0, y0 = points_norm_new[0]
    canvas.coords(text_id, x0 * canvas_width + 4, y0 * canvas_height - 2)
    boxes[idx] = (cls, points_norm_new, poly_id, text_id)
    draw_selection_handle(idx)

def load_image(idx):
    global img_tk, boxes
    cancel_obb()
    deselect()
    boxes = []
    canvas.delete("all")

    img_path = os.path.join(image_dir, image_files[idx])
    image = Image.open(img_path).resize((canvas_width, canvas_height))
    img_tk = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    root.title(f"YOLO-OBB Label Editor — {image_files[idx]}  ({idx + 1}/{len(image_files)})")

    name = os.path.splitext(image_files[idx])[0]
    label_path = os.path.join(label_dir, f"{name}.txt")
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 9:
                    cls = int(float(parts[0]))
                    coords = list(map(float, parts[1:]))
                    points_norm = list(zip(coords[0::2], coords[1::2]))
                    poly_id, text_id = draw_polygon_on_canvas(cls, points_norm)
                    boxes.append((cls, points_norm, poly_id, text_id))

def place_obb_click(x, y):
    if len(obb_points) < 2:
        obb_points.append((x, y))
        marker = canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill='yellow', outline='black')
        obb_point_markers.append(marker)
        if len(obb_points) == 2:
            global rubber_line, obb_edge_line
            if rubber_line is not None:
                canvas.delete(rubber_line)
                rubber_line = None
            a, b = obb_points
            obb_edge_line = canvas.create_line(a[0], a[1], b[0], b[1], fill='yellow', width=2)
    else:
        finish_obb((x, y))

def on_left_press(event):
    global dragging_handle
    x, y = event.x, event.y

    if handle_hit(x, y):
        dragging_handle = True
        return

    if obb_points:
        place_obb_click(x, y)
        return

    hit_idx, _ = find_box_at(x, y)
    if hit_idx is not None:
        select_box(hit_idx)
        return

    if selected_idx is not None:
        deselect()
        return

    place_obb_click(x, y)

def on_left_drag(event):
    if dragging_handle and selected_idx is not None:
        rotate_selected_to(event.x, event.y)

def on_left_release(event):
    global dragging_handle
    dragging_handle = False

def on_mouse_move(event):
    global rubber_line, preview_shape
    x, y = event.x, event.y

    if len(obb_points) == 1:
        a = obb_points[0]
        if rubber_line is None:
            rubber_line = canvas.create_line(a[0], a[1], x, y, fill='yellow', dash=(4, 2))
        else:
            canvas.coords(rubber_line, a[0], a[1], x, y)
    elif len(obb_points) == 2:
        a, b = obb_points
        corners = compute_obb_corners(a, b, (x, y))
        if corners is None:
            return
        coords = []
        for cx, cy in corners:
            coords.extend([cx, cy])
        if preview_shape is None:
            preview_shape = canvas.create_polygon(coords, outline='yellow', fill='', width=1, dash=(4, 2))
        else:
            canvas.coords(preview_shape, *coords)

def find_box_at(x, y):
    for i, box in reversed(list(enumerate(boxes))):
        points_px = [(px * canvas_width, py * canvas_height) for px, py in box[1]]
        if point_in_polygon(x, y, points_px):
            return i, box
    return None, None

def delete_box_at(x, y):
    global selected_idx
    i, box = find_box_at(x, y)
    if box is not None:
        canvas.delete(box[2])
        canvas.delete(box[3])
        boxes.pop(i)
        if selected_idx is not None:
            if selected_idx == i:
                deselect()
            elif selected_idx > i:
                selected_idx -= 1
        return True
    return False

def on_right_click(event):
    if obb_points:
        undo_obb_point()
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
    cancel_obb()
    deselect()
    save_label()
    if index < len(image_files) - 1:
        index += 1
        load_image(index)

def prev_image():
    global index
    cancel_obb()
    deselect()
    save_label()
    if index > 0:
        index -= 1
        load_image(index)

def clear_boxes():
    global boxes
    cancel_obb()
    deselect()
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

canvas.bind("<ButtonPress-1>", on_left_press)
canvas.bind("<B1-Motion>", on_left_drag)
canvas.bind("<ButtonRelease-1>", on_left_release)
canvas.bind("<Motion>", on_mouse_move)
canvas.bind("<ButtonPress-3>", on_right_click)
canvas.bind("<ButtonPress-2>", on_middle_click)

def on_escape(event=None):
    cancel_obb()
    deselect()

root.bind("<Escape>", on_escape)

tk.Label(root, text="Left-click 3x on empty canvas: draw new box  |  Left-click a box: select, "
                     "then drag its cyan handle to rotate  |  Esc: cancel/deselect  |  "
                     "Right-click: undo click / delete  |  Middle-click: relabel",
         font=('Arial', 9), fg='gray').pack(pady=(0, 4))

# === START ===
load_image(index)
root.mainloop()
