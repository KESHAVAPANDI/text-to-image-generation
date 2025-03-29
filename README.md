# text-to-image-generation
a stable diffusion model that is combined with LCM LoRA to reduce >50 steps to &lt;10 steps
In the s5(Offline) folder you need to manually install those models and the app.py will combine those models
In the s5 folder it directly fetches the model from huggingface and the app.py combines the model

IMPORTANT:- 
pytorch version : 2.6.0 + cuda 12.5
(cuda in the 12... version)
the safetensors will install for only once for initialising
