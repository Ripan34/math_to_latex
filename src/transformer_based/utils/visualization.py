import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(img, attn, token):
    attn_map = attn.reshape(int(np.sqrt(len(attn))), -1)
    plt.imshow(img, cmap='gray')
    plt.imshow(attn_map, cmap='jet', alpha=0.4)
    plt.title(f"Attention for {token}")
    plt.show()