configs = {
    'train_from_scratch': 1,

    "train_path": r"/home/lishijia/DeepPacker/dataset/deeppacker/train.pkl",
    "val_path": r"/home/lishijia/DeepPacker/dataset/deeppacker/val.pkl",
    "vocab_path": r"D:/test623/PackerBert/dataset/vocab.txt",
    "vocab_size": 2672,
    "train_size": 320000,
    "num_classes": 2,
    "pretrain_model_save_path": r'D:/test623/PackerBert/checkpoint/ep4.pt',
    "current_model_save_path": r'/home/lishijia/DeepPacker/PackerLM/checkpoints/task1/checkpoint_current.pt',
    "best_model_save_path": r'/home/lishijia/DeepPacker/PackerLM/checkpoints/task1/checkpoint_best.pt',

    "visualizer_path": r'/home/lishijia/DeepPacker/PackerLM/visualizer/task1',

    "hidden": 768,  # bert hidden, i.e. embedding dimension
    "n_layers": 8,  # how many transformer encoder layers
    "attn_heads": 8,  # how many heads in multi-head attention

    "num_workers": 16,  # data loader num_workers
    "max_len": 512,  # how many tokens in one sample
    "batch_size": 32,  # how many samples in one batch
    "epochs": 10,
    "lr": 1e-5,
    "betas": (0.9, 0.999),
    "weight_decay": 0.01,
    "warmup_ratio": 0.01,
    "with_cuda": True,  # if forward and backward in GPU
    "cuda_devices": None,  # how many GPUs you want use
    "log_freq": 10
}
