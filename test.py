from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("model_fold/Illustrious-XL-v0.1/Illustrious-XL-v0.1")



pipe.load_lora_weights("model_fold/Illustrious-XL-v0.1/yinlin/Yinlinx IL v22.safetensors")

prompt = "yinlinx,1girl, red hair, ponytail, purple eyes,  circle facial mark, bikini, full body,  looking at viewer, calm expression, soft eyes, hairband, single earring, hair stick, solo, closed mouth, standing in a swimming pool"

image = pipe(prompt=prompt).images[0]

image.save("output.png")
