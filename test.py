from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

def main():
    # Load model and processor
    model_name = "ShaoRun/RS-EoT-7B"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="cuda:1"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    # Define input image (assumes demo.jpg is in the current directory)
    image_path = "./assets/demo.jpg" 

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "How many cars in this image?"},
            ],
        }
    ]

if __name__ == "__main__":
    main()