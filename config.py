configs = {
    'train_from_scratch': 1,

    "train_path": r"E:/DeepPackerDataset/pretrain_compare/dataset_shuffled",
    "val_path": r"E:/DeepPacker/dataset/pretrain/val",
    "vocab_path": r'E:/DeepPacker/PackerLM/PackerBert/dataset/vocab.txt',
    "vocab_size": 2672,
    # "train_size_124": 10164119,
    "train_size": 1610547,
    "val_size": 1007689,

    "current_model_save_path": r'E:/DeepPacker/PackerLM/checkpoints/checkpoint_current.pt',
    "best_model_save_path": r'E:/DeepPacker/PackerLM/checkpoints/checkpoint_best.pt',

    "visualizer_path": r'E:/DeepPacker/PackerLM/visualizer/mlm_task',

    "hidden": 768,  # bert hidden, i.e. embedding dimension
    "n_layers": 8,  # how many transformer encoder layers
    "attn_heads": 8,  # how many heads in multi-head attention

    "num_workers": 16,  # data loader num_workers
    "max_len": 512,  # how many tokens in one sample
    "batch_size": 64,  # how many samples in one batch
    "epochs": 5,
    "lr": 1e-6,
    "betas": (0.9, 0.999),
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "with_cuda": True,  # if forward and backward in GPU
    "cuda_devices": None,  # how many GPUs you want use
    "log_freq": 10
}
