import torch
from diffusers import LattePipeline

pipeline = LattePipeline.from_pretrained(
	"maxin-cn/Latte-1", torch_dtype=torch.float16
).to("cuda")

pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.vae.to(memory_format=torch.channels_last)

pipeline.transformer = torch.compile(pipeline.transformer)
pipeline.vae.decode = torch.compile(pipeline.vae.decode)

video = pipeline(prompt="A dog wearing sunglasses floating in space, surreal, nebulae in background").frames[0]