import torch
from PIL import Image
from tinychart.model.builder import load_pretrained_model
from tinychart.mm_utils import get_model_name_from_path
from tinychart.eval.run_tiny_chart import inference_model
from tinychart.eval.eval_metric import parse_model_output, evaluate_cmds

# Build the model
model_path = "/lab-share/CHIP-Majumder-PNR-e2/Public/thomas/mPLUG-DocOwl/TinyChart/TinyChart-3B-768"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device="cuda",  # device="cpu" if running on cpu
)

img_path = "/lab-share/CHIP-Majumder-PNR-e2/Public/thomas/vlm-ensemble-chart-extraction/data/ChartQA Dataset/test/png/166.png"
text = "Generate underlying data table for the chart."
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
