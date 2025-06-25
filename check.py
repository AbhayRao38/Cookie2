import numpy as np

probs = np.load(r"C:\Users\praoa\OneDrive\Desktop\Projects\multimodal-emotion-mci-system\model_eye\eye_softmax_probs.npy")  # replace with actual file path
print(probs.shape)
print(probs[:5])  # print first 5 predictions for preview
