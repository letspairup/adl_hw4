import os
import json
from pathlib import Path
from homework.generate_qa import generate_qa_pairs

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "train"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "train" / "generated_qa_pairs.json"

def main():
    all_qas = []
    for info_file in DATA_DIR.glob("*_info.json"):
        frame_base = info_file.stem.replace("_info", "")
        for view_index in range(10):  # views 0 to 9
            image_path = DATA_DIR / f"{frame_base}_{view_index:02d}_im.jpg"
            if not image_path.exists():
                continue
            try:
                qas = generate_qa_pairs(str(info_file), view_index)
                for qa in qas:
                    qa["image_file"] = f"train/{frame_base}_{view_index:02d}_im.jpg"
                    all_qas.append(qa)
            except Exception as e:
                print(f"Failed on {info_file} view {view_index}: {e}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_qas, f, indent=2)

    print(f"✅ Generated {len(all_qas)} QA pairs to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
