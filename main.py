from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "model_fold/stable-diffusion-xl-base-1.0/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe.to("cuda")


pipe.load_lora_weights("model_fold/yinlin_lora_model/pytorch_lora_weights.safetensors")




prompt = "<yinlin>, 1girl, (masterpiece:1.2), (best quality:1.2), mole_under_eye, pointy_ears, purple_eyes, red_hair , black_hairband, standing on ground, white background,sharp focus, super fine illustration,vibrant colors, cinematic lighting,ultra detailed, smile"

image = pipe(prompt=prompt).images[0]

image.save("output.png")
