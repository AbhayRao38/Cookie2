import numpy as np

paths = {
    'face': r'C:\Users\praoa\OneDrive\Desktop\Projects\multimodal-emotion-mci-system\model_face\X_face_test_96x96.npy',
    'eye': r'C:\Users\praoa\OneDrive\Desktop\Projects\multimodal-emotion-mci-system\model_eye\model_eye\X_eye_test_idx.npy',
    'speech': r'C:\Users\praoa\OneDrive\Desktop\Projects\multimodal-emotion-mci-system\model_speech\X_test.npy',
    'mri': r'C:\Users\praoa\OneDrive\Desktop\Projects\multimodal-emotion-mci-system\model_mri\X_mri_test_idx.npy',
}

for name, path in paths.items():
    idx = np.load(path)
    print(f"{name.upper()} indices shape: {idx.shape}, min: {idx.min()}, max: {idx.max()}")
