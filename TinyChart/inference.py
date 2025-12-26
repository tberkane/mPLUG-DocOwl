import os
import sys
import json
import torch
from PIL import Image
from tinychart.model.builder import load_pretrained_model
from tinychart.mm_utils import get_model_name_from_path
from tinychart.eval.run_tiny_chart import inference_model


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <image_dir> <output_json>")
        sys.exit(1)

    image_dir = sys.argv[1]
    output_json = sys.argv[2]

    # Build the model
    model_path = "mPLUG/TinyChart-3B-768"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device="cuda",  # Change to "cpu" if running on cpu
    )

    # Get all image files in the directory (common image extensions)
    image_extensions = (".png", ".jpg", ".jpeg", ".gif")
    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith(image_extensions)
        and os.path.isfile(os.path.join(image_dir, f))
    ]
    image_files.sort()

    # If output file exists, load existing results to avoid duplicates
    results = []
    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            try:
                results = json.load(f)
            except Exception:
                results = []
        done_images = set(r["imagename"] for r in results if "imagename" in r)
    else:
        done_images = set()

    # Open output file in append mode
    with open(output_json, "a", encoding="utf-8") as out_f:
        for img_name in image_files:
            if img_name in done_images:
                continue
            img_path = os.path.join(image_dir, img_name)
            text = "Generate underlying data table for the chart."
            try:
                response = inference_model(
                    [img_path],
                    text,
                    model,
                    tokenizer,
                    image_processor,
                    context_len,
                    conv_mode="phi",
                    max_new_tokens=1024,
                )
            except Exception as e:
                response = f"Error: {str(e)}"
            result = {"imagename": img_name, "answer": response}
            # Append to file as a single JSON object per line, then reformat at end if needed
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

    # After all done, reformat file as a JSON list (if not already)
    # Read all lines, parse as JSON, write as a list
    with open(output_json, "r", encoding="utf-8") as f:
        lines = f.readlines()
    objs = [json.loads(line) for line in lines if line.strip()]
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(objs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
