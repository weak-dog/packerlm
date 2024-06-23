import shutil
import os
import torch
from config_classify import configs
from PackerBert.model.bert_framework import BERT
from PackerBert.model.classify import CLASSIFIER
from PackerBert.trainer.train_classify import ClassifyTrainer
from PackerBert.dataset.AsmVocab import AsmVocab
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

if __name__ == '__main__':

    train_from_scratch = configs['train_from_scratch']

    train_path = configs['train_path']
    val_path = configs['val_path']
    vocab_path = configs["vocab_path"]
    vocab_size = configs["vocab_size"]
    train_size = configs["train_size"]

    pretrain_model_save_path = configs["pretrain_model_save_path"]
    current_model_save_path = configs['current_model_save_path']
    best_model_save_path = configs["best_model_save_path"]

    visualizer_path = configs["visualizer_path"]

    # bert config
    hidden = configs["hidden"]
    n_layers = configs["n_layers"]
    attn_heads = configs["attn_heads"]

    num_workers = configs["num_workers"]
    max_len = configs["max_len"]
    batch_size = configs["batch_size"]
    epochs = configs["epochs"]
    lr = configs["lr"]
    betas = configs["betas"]
    weight_decay = configs["weight_decay"]
    warmup_ratio = configs["warmup_ratio"]
    with_cuda = configs["with_cuda"]
    cuda_devices = configs["cuda_devices"]
    log_freq = configs["log_freq"]

    vocab = AsmVocab()
    vocab.load(vocab_path)

    logger.info(f"Bert config: hidden:{hidden}, n_layers:{n_layers}, attn_heads:{attn_heads}")
    logger.info(f"Dataset config:  max_len:{max_len}, batch_size:{batch_size} ")
    logger.info(f"Train config: epochs:{epochs}, initial lr:{lr}, betas:{betas}, weight_decay:{weight_decay}, "
                f"warmup_ratio:{warmup_ratio}, with_cuda:{with_cuda}, cuda_devices:{cuda_devices},")
    logger.info(f"Log config: log_freq:{log_freq}")
    logger.info(f"Vocab size:{vocab_size}")

    torch.cuda.set_device(0)

    bert = BERT(vocab_size, hidden=hidden, n_layers=n_layers, attn_heads=attn_heads).to("cuda:0")
    model = CLASSIFIER(bert).to("cuda:0")

    logger.info("Initializing Bert Language Model model Done!")

    if train_from_scratch:
        if os.path.exists(visualizer_path):
            shutil.rmtree(visualizer_path)
        if os.path.exists(current_model_save_path):
            os.remove(current_model_save_path)
        if os.path.exists(best_model_save_path):
            os.remove(best_model_save_path)
        checkpoint = torch.load(pretrain_model_save_path, map_location="cuda:0")
        model.bert.load_state_dict(checkpoint)
    else:
        checkpoint = torch.load(current_model_save_path, map_location="cuda:0")
        model.load_state_dict(checkpoint["model_state_dict"])

    visualizer = SummaryWriter(visualizer_path)
    logger.info(f"Initializing Visualizer Done! Visualizer_path:{visualizer_path}")

    trainer = ClassifyTrainer(model, vocab,
                              current_model_save_path=current_model_save_path,
                              best_model_save_path=best_model_save_path,
                              train_path=train_path, val_path=val_path,
                              num_workers=num_workers,
                              max_len=max_len, batch_size=batch_size,
                              train_size=train_size,
                              visualizer=visualizer, epochs=epochs,
                              lr=lr, betas=betas, weight_decay=weight_decay,
                              warmup_ratio=warmup_ratio, with_cuda=with_cuda, cuda_devices=cuda_devices,
                              log_freq=log_freq, random_seed=66)

    if not train_from_scratch:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['current_epoch']
        trainer.current_dataloader_idx = checkpoint['current_dataloader_idx']
        trainer.checkpoint_best_score = checkpoint['checkpoint_best_score']
        trainer.all_predictions = checkpoint['all_predictions']
        trainer.all_labels = checkpoint['all_labels']
        trainer.sum_loss = checkpoint["sum_loss"]

    logger.info("Initializing Classifier Trainer Done!")
    print("current epoch:", trainer.current_epoch)
    print("current dataloader idx:", trainer.current_dataloader_idx)
    print("sum_loss:", trainer.sum_loss)
    print("all_predictions", len(trainer.all_predictions))

    trainer.classify_train()
    visualizer.close()

