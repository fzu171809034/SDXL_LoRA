from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("model_fold/Illustrious-XL-v0.1/Illustrious-XL-v0.1",
                                         torch_dtype=torch.float16).to("cuda")


pipe.load_lora_weights("model_fold/Illustrious-XL-v0.1/yinlin/Yinlinx_IL_v22.safetensors", adapter_name="yinlin")
pipe.load_lora_weights("model_fold/Illustrious-XL-v0.1/Stabilizer/TrendCraft_The_Peoples_Style_Detailer-v1.0A-3_13_2025-illustrious.safetensors", adapter_name="style")

pipe.set_adapters(["yinlin", "style"], adapter_weights=[1.0, 0.7])

pipe.to(torch.float16)

prompt = ("masterpiece,best quality,amazing quality,"
        "1girl, yinlinx, echoset1 outfit, purple eyes,long hair, ponytail, red hair, circle facial mark, hairband, full body, floating hair, smile, relaxed eyes, hand up, standing, dynamic pose, swirling energy, light waves, sparkling particles,Starry sky background, ambient light,"
          )
neg_prompt = "lowres, (worst quality, bad quality:1.2), bad anatomy, sketch, jpeg artifacts, deformed, bad hand, patreon username, web address, signature, watermark, text, logo, artist name, half body"

image = pipe(prompt=prompt,negative_prompt=neg_prompt).images[0]

image.save("output.png")
