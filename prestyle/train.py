import torch, os
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from omegaconf import OmegaConf
import importlib
from pytorch_lightning.callbacks import ModelCheckpoint

# torch.autograd.set_detect_anomaly(True)    # NaN/Inf 추적
torch.set_float32_matmul_precision("medium")   # 텐서코어 사용
torch.backends.cudnn.benchmark = True          # Conv/BN autotune

def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def create_data(config):
    data_cls = get_obj_from_str(config.target)
    data = data_cls(data_config=config)
    return data


def get_model(cfgs):
    model = instantiate_from_config(cfgs.model)
    # if "load_ckpt_path" in cfgs:
    #     model.load_state_dict(torch.load(cfgs.load_ckpt_path, map_location="cpu")["state_dict"], strict=False) # ckpt 두번 안데려오게 제거
    return model


def train(cfgs):
    # create model
    model = get_model(cfgs)

    # create data
    data_config = cfgs.pop("data", OmegaConf.create())
    data_opt = data_config
    data = create_data(data_opt)
    data.prepare_data()
    data.setup(stage='fit')

    checkpoint_callback = ModelCheckpoint(every_n_epochs=cfgs.check_freq, save_top_k=-1)
    imagelogger_callback = instantiate_from_config(cfgs.imagelogger_callback)

    # 3) ⬇️ CSVLogger 추가  (logs_csv/lightning_logs/version_*/metrics.csv 로 저장)
    csv_logger = CSVLogger("logs_csv")
    tb_logger   = TensorBoardLogger("logs_tb")        # ★ 추가

    trainer = pl.Trainer(
        # detect_anomaly=True,          # ← NaN 시 stack trace
        gradient_clip_val=1.0,        # ← 폭발적 그래디언트 억제
        resume_from_checkpoint=cfgs.load_ckpt_path,
        logger=[tb_logger, csv_logger],
        callbacks=[checkpoint_callback, imagelogger_callback], 
        **cfgs.lightning
    )
    trainer.fit(model=model, datamodule=data)

if __name__ == "__main__":
    config_path = 'configs/StyleTrain.yaml'
    cfgs = OmegaConf.load(config_path)
    train(cfgs)