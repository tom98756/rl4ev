import torch

from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from lightning.pytorch.loggers import WandbLogger


from rl4co.utils.trainer import RL4COTrainer

from evrp.env import EVREnv
from evrp.model import Model

def main():

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create Env instance
    env = EVREnv()

    model = Model(
        env,
        baseline="rollout",
        dataloader_num_workers=4,
        batch_size=512,
        train_data_size=1_000,
        val_data_size=1_0,
        optimizer_kwargs={'lr': 1e-4}
    )

    # Example callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",  # save to checkpoints/
        filename="epoch_{epoch:03d}",  # 保存为 epoch_XXX_YYYY-MM-DD.ckpt
        save_top_k=1,  # save only the best model
        save_last=True,  # save the last model
        monitor="val/reward",  # monitor validation reward
        mode="max",
    )  # maximize validation reward
    rich_model_summary = RichModelSummary(max_depth=3)  # model summary callback
    callbacks = [checkpoint_callback, rich_model_summary]

    # Logger
    logger = WandbLogger(project="rl4co", name="ev")
    # logger = None # uncomment this line if you don't want logging

    # Main trainer configuration
    trainer = RL4COTrainer(
        max_epochs=10,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=callbacks
    )

    # Main training loop
    trainer.fit(model)


if __name__ == "__main__":
    main()