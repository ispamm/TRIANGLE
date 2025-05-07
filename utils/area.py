import torch
import numpy as np

# THIS IS THE CORE PY CODE OF TRIANGLE FRAMEWORK



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