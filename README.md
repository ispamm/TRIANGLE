<h2 align="center"> A TRIANGLE Enables Multimodal Alignment Beyond
Cosine Similarity </a></h2>



<h5 align="center"> 
<h5 align="center">
     


### ✨ Takeaway functions 

`triangle_computation`

This function computes the area of the k-dimensional parallelotope formed by three vectors—one from each modality—using their triangle matrix determinant:

```python
def basic_area_computation(language, video, audio):
    #Assume input vectors each of shape [1,latent_dim]
    #Assume they are all normalized to have norm=1

    u = language - video  # Shape: (1, dim)
    v = language - audio # Shape: (1, dim)

    # Compute the norms for u and v
    u_norm = u @ u.T # Shape: (1)
    v_norm = v @ v.T  # Shape: (1)

    # Compute the dot products 
    uv_dot = u @ v.T 


    # Calculate the area. I remove sqrt calculation
    area = torch.sqrt((u_norm * v_norm) - (uv_dot ** 2)) / 2  # Shape: (n, n)
    
    return area
```


This simple geometric operation scales to batches and more complex setups in the full TRIANGLE function below.

`area_computation`


```python
def area_computation(language, video, audio):


    #print(f"norm language= {torch.sum(language ** 2, dim=1)}")
    
    language_expanded = language.unsqueeze(1)  # Shape: (n, 1, dim)

    # Compute the differences for all pairs (i-th language embedding with all j-th video/audio embeddings)
    u = language_expanded - video.unsqueeze(0)  # Shape: (n, n, dim)
    v = language_expanded - audio.unsqueeze(0)  # Shape: (n, n, dim)

    # Compute the norms for u and v
    u_norm = torch.sum(u ** 2, dim=2)  # Shape: (n, n)
    v_norm = torch.sum(v ** 2, dim=2)  # Shape: (n, n)

    # Compute the dot products for all pairs
    uv_dot = torch.sum(u * v, dim=2)  # Shape: (n, n)

    # Calculate the area for all pairs. I remove sqrt calculation
    area = ((u_norm * v_norm) - (uv_dot ** 2))/2#torch.sqrt((u_norm * v_norm) - (uv_dot ** 2)) / 2  # Shape: (n, n)
    
    return area
```

### 🧐 how to use it in practice?  Implementation of the InfoNCE loss with area:

```python
import torch
import torch.nn.functional as F

# Hyperparameters
bs = 32
latent_dim = 512
contrastive_temp = 0.07

# Output of the encoders
language = torch.randn((bs,latent_dim))
video = torch.randn((bs,latent_dim))
audio = torch.randn((bs,latent_dim))

area = area_computation(language,video,audio)
area = area / contrastive_temp


areaT = area_computation(language,video,audio).T
areaT = areaT / contrastive_temp

targets = torch.linspace(0, bs - 1, bs, dtype=int)

loss = (
        F.cross_entropy(-area, targets, label_smoothing=0.1) #d2a
        + F.cross_entropy(-areaT, targets, label_smoothing=0.1) #a2d
) / 2

print(loss)

```


## Building Environment
TRIANGLE is implemented based on Pytorch. We use Python-3.9 and Cuda-11.7. Other version could be also compatible. Other needed packages are listed in preinstall.sh.

```
conda create -n triangle python=3.9
conda activate triangle
sh preinstall.sh
```

## Download basic encoder's pretrained checkpoints
Make a dir named pretrained_weights under the main work dir.

1. Download evaclip weight:
```
wget -P pretrained_weights/clip/ https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA01_CLIP_g_14_psz14_s11B.pt
```
2. Download beats weight from https://github.com/microsoft/unilm/tree/master/beats

3. Download bert weight:
```python
from transformers import BertModel, BertTokenizer
bert = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert.save_pretrained('pretrained_weights/bert/bert-base-uncased')
bert_tokenizer.save_pretrained('pretrained_weights/bert/bert-base-uncased')
```


The processed  pretrained_weights path should be as follows:
```
    ├── pretrained_weights
    │   ├── beats
    │   │   └── BEATs_iter3_plus_AS2M.pt
    │   ├── bert
    │   │   └── bert-base-uncased
    │   ├── clip
    │   │   └── EVA01_CLIP_g_14_psz14_s11B.pt
```


## MODEL ZOO

Model pretrained on VAST subset will be available after publication to preserve anonimity.






## Download  VAST-27M annotations for pretraining

VAST-27M DATASET could be downloaded following the official [repo](https://github.com/TXH-mercury/VAST)

We used a subset of VAST-27M for the pretraining phase of TRIANGLE. Will be available after publication to preserve anonimity.


## Finetune  Model on the 150k subset of VAST27M
Download annotations150k.json file subset.
Reference it in scripts/triangle/finetune_ret.sh and in config/triangle/finetune_cfg/finetune-area.json
```
sh scripts/triangle/finetune_ret.sh
```


## Finetune  Model on downstream datasets
Change configuration internally at scripts/triangle/finetune_ret.sh and then run

```
sh scripts/triangle/finetune_ret.sh
```




## Test your finetuned Model
For example, if the cmd for finetuning retrieval model is as follows:

```
python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9834 \
./run.py \
--learning_rate 2e-5 \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/triangle/finetune_cfg/retrieval-msrvtt.json \
--pretrain_dir $PATH-TO-CKPT-FOLDER \
--output_dir $PATH-WHERE-TO-STORE-RESULTS \
```

if you want to test model, just add following two rows to the cmd:
```
--mode 'testing' \
--checkpoint /PATH/TO/SAVED_CHECKPOINT.pt
```



## Third-Party Licenses

For the full list of third-party licenses used in this project, please see the [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) file.
