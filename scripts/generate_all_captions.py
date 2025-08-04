from pathlib import Path
import json
from homework.generate_captions import generate_caption

def main():
    data_dir = Path(__file__).resolve().parent.parent / "data" / "train"
    output_file = data_dir / "generated_captions.json"

    all_captions = []

    for info_path in sorted(data_dir.glob("*_info.json")):
        base = info_path.stem.replace("_info", "")
        for view_index in range(10):
            try:
                captions = generate_caption(str(info_path), view_index)
                image_file = f"train/{base}_{view_index:02d}_im.jpg"
                for caption in captions:
                    all_captions.append({
                        "image_file": image_file,
                        "caption": caption
                    })
            except Exception as e:
                print(f"⚠️ Skipped {info_path.name} view {view_index}: {e}")

    with open(output_file, "w") as f:
        json.dump(all_captions, f, indent=2)

    print(f"✅ Generated {len(all_captions)} captions → {output_file}")

if __name__ == "__main__":
    main()
