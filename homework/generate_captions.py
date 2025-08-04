from pathlib import Path

import fire
from matplotlib import pyplot as plt
from .generate_qa import extract_kart_objects, draw_detections, extract_track_info, extract_frame_info

def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate captions for a specific view as a list of strings.
    """
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track = extract_track_info(info_path)

    if not karts:
        return []

    ego = next((k for k in karts if k["is_center_kart"]), None)
    if ego is None:
        return []

    ego_name = ego["kart_name"]
    num_others = len(karts) - 1

    captions = []

    # 1. Ego
    captions.append(f"{ego_name} is the ego car.")

    # 2. Count
    captions.append(f"There are {len(karts)} karts in the scenario.")

    # 3. Track
    captions.append(f"The track is {track}.")

    # 4. Relative positions
    ego_x, ego_y = ego["center"]
    for kart in karts:
        if kart["instance_id"] == ego["instance_id"]:
            continue
        name = kart["kart_name"]
        x, y = kart["center"]
        dx, dy = x - ego_x, y - ego_y
        horiz = "left" if dx < -5 else "right" if dx > 5 else None
        vert = "front" if dy < -5 else "back" if dy > 5 else None
        if horiz and vert:
            pos = f"{vert} and {horiz}"
        elif vert:
            pos = vert
        elif horiz:
            pos = horiz
        else:
            pos = "near"
        captions.append(f"{name} is {pos} of the ego car.")

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption})


if __name__ == "__main__":
    main()
