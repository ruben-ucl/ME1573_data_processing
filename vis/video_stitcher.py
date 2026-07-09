import subprocess

video1 = r"F:\AlSi10Mg single layer ffc\videos\bs-f40\0514_04_bs-f40.mp4"
video2 = r"F:\AlSi10Mg single layer ffc\videos\bs-f40\0514_04_bs-f40.mp4"
video3 = r"F:\ESRF ME1573 LTP 6 Al data HDF5\ffc\videos\bs-f40\1112_01_bs-f40.mp4"  # 768x168
video4 = r"F:\sim_segmented_300W_800mm_s\SLM_Al10SiMg_1st_layer_4mm_300W_800mms_with_phases\animations\centre_streamlines_2.mp4"  # 2310x1082
out    = r"vis/stitched_video_3pane.mp4"

width = 1080

# Crop fractions: (crop_from_top, crop_from_bottom)
crop1 = (0.25, 0.35)
crop2 = (0.9, 0.0)
crop3 = (0.0, 0.0)
crop4 = (0.0, 0.20)

# Number of times each video loops (1 = no loop, 2 = play twice, etc.)
loops1 = 3
loops2 = 3
loops3 = 1
loops4 = 1

# Annotation text in top-left corner (None for no annotation)
annotation1 = None
annotation2 = '40,000 fps'
annotation3 = '504,000 fps'
annotation4 = 'Simulation'

# Annotation style
font_file  = r"C:/Users/lbn38569/AppData/Local/anaconda3/envs/ml/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"
font_size  = 30
box_color  = "0x00000000"   ## background box, use "0x00000000" for no background
padding    = 10             # pixels between text and corner / box edge

# Annotation position options: "top_left", "top_right", "bottom_left", "bottom_right", "top_centre", "bottom_centre"
position1 = "bottom_centre"
position2 = "bottom_centre"
position3 = "bottom_centre"
position4 = "top_centre"

# Per-annotation font colour
font_color1 = "white"
font_color2 = "white"
font_color3 = "white"
font_color4 = "black"

# Dividing line beneath each video
line_after_1 = False
line_after_2 = True
line_after_3 = False
line_after_4 = False

line_thickness = 6
line_color     = "white"

# --- Helpers ---
def crop_filter(crop_top_frac, crop_bot_frac):
    if crop_top_frac == 0.0 and crop_bot_frac == 0.0:
        return ""
    top_px = f"ih*{crop_top_frac}"
    new_h  = f"ih*{1.0 - crop_top_frac - crop_bot_frac}"
    return f"crop=iw:{new_h}:0:{top_px}"

def annotation_filter(text, position, font_file, font_size, font_color, box_color, padding):
    if not text:
        return ""

    if box_color == "0x00000000":
        box_str = "box=0"
    else:
        box_str = f"box=1:boxcolor={box_color}:boxborderw={padding}"

    positions = {
        "top_left":      (f"{padding}",                f"{padding}"),
        "top_centre":    (f"(w-text_w)/2",             f"{padding}"),
        "top_right":     (f"w-text_w-{padding}",       f"{padding}"),
        "bottom_left":   (f"{padding}",                f"h-text_h-{padding}"),
        "bottom_centre": (f"(w-text_w)/2",             f"h-text_h-{padding}"),
        "bottom_right":  (f"w-text_w-{padding}",       f"h-text_h-{padding}"),
    }
    x, y = positions.get(position, positions["top_left"])

    # Escape colon in font path (e.g. C:/...) and comma in text
    escaped_font = font_file.replace("\\", "/").replace(":", "\\:")
    escaped_text = text.replace(",", "\\,")

    return (
        f"drawtext=text='{escaped_text}'"
        f":fontfile='{escaped_font}'"
        f":fontsize={font_size}"
        f":fontcolor={font_color}"
        f":{box_str}"
        f":x={x}:y={y}"
    )
    
def build_filter(crops, loops, annotations, font_colors, positions, lines, width,
                 font_file, font_size, box_color, padding,
                 line_thickness, line_color):
    parts  = []
    labels = []

    for i, (crop_t, crop_b) in enumerate(crops):
        # Loop
        if loops[i] > 1:
            loop = f"[{i}:v]loop={loops[i] - 1}:size=32767:start=0"
            loop_label = f"[vlp{i}]"
            parts.append(f"{loop}{loop_label}")
            src = loop_label
        else:
            src = f"[{i}:v]"

        # Scale
        scale = f"{src}scale={width}:-2"
        c     = crop_filter(crop_t, crop_b)
        chain = f"{scale},{c}" if c else scale
        label = f"[v{i}]"
        parts.append(f"{chain}{label}")
        current_label = label

        # Annotation
        a = annotation_filter(annotations[i], positions[i], font_file, font_size, font_colors[i], box_color, padding)
        if a:
            ann_label = f"[va{i}]"
            parts.append(f"{current_label}{a}{ann_label}")
            current_label = ann_label

        # Dividing line
        if lines[i]:
            lined_label = f"[vl{i}]"
            parts.append(
                f"{current_label}drawbox=x=0:y=ih-{line_thickness}:"
                f"w=iw:h={line_thickness}:color={line_color}:t=fill{lined_label}"
            )
            current_label = lined_label

        labels.append(current_label)

    stack = "".join(labels) + f"vstack=inputs={len(labels)}[v]"
    parts.append(stack)
    return ";".join(parts)


# --- Build & run ---
crops       = [crop1, crop2, crop3, crop4]
loops       = [loops1, loops2, loops3, loops4]
annotations = [annotation1, annotation2, annotation3, annotation4]
font_colors = [font_color1, font_color2, font_color3, font_color4]
lines       = [line_after_1, line_after_2, line_after_3, line_after_4]
positions   = [position1, position2, position3, position4]

filter_complex = build_filter(
    crops, loops, annotations, font_colors, positions, lines, width,
    font_file, font_size, box_color, padding,
    line_thickness, line_color
)

subprocess.run([
    "ffmpeg",
    "-i", video1,
    "-i", video2,
    "-i", video3,
    "-i", video4,
    "-filter_complex", filter_complex,
    "-map", "[v]",
    out
])