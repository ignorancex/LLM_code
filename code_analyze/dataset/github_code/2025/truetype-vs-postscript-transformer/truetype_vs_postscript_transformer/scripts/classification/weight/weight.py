"""Main script for training the FontClassifier model."""

from typing import Literal

import pytorch_lightning as pl
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)

from truetype_vs_postscript_transformer.lightning.data_modules import MultiFontLDM
from truetype_vs_postscript_transformer.lightning.modules import StyleClassifierLM


def weight_classifier(
    outline_mode: Literal[
        "truetype_point",
        "truetype_atomic_point",
        "truetype_segment",
        "postscript_segment",
    ],
) -> None:
    """Train the FontClassifier model."""
    variable_fonts = [
        TTFont("./fonts/ofl/notosansjp/NotoSansJP[wght].ttf"),
        TTFont("./fonts/ofl/notoserifjp/NotoSerifJP[wght].ttf"),
        TTFont("./fonts/ofl/mplus1/MPLUS1[wght].ttf"),
    ]
    weight_values = [300, 400, 500, 700]
    instantiated_variable_fonts = [
        instantiateVariableFont(
            variable_font,
            {"wght": weight},
            inplace=False,
            updateFontNames=True,
        )
        for variable_font in variable_fonts
        for weight in weight_values
    ]
    fonts = [
        *instantiated_variable_fonts,
        TTFont("./fonts/ofl/roundedmplus1c/RoundedMplus1c-Light.ttf"),
        TTFont("./fonts/ofl/roundedmplus1c/RoundedMplus1c-Regular.ttf"),
        TTFont("./fonts/ofl/roundedmplus1c/RoundedMplus1c-Medium.ttf"),
        TTFont("./fonts/ofl/roundedmplus1c/RoundedMplus1c-Bold.ttf"),
    ]

    class_labels = [
        f"{font['name'].getBestFamilyName()} {font['name'].getBestSubFamilyName()}"
        for font in fonts
    ]

    data_module = MultiFontLDM(
        fonts=fonts,
        outline_mode=outline_mode,
        batch_size=1024,
        split_ratios=(0.8, 0.1, 0.1),
        seed=33114113,
        pad_size=None,
    )

    if outline_mode in ["truetype_point", "truetype_atomic_point"]:
        model_name = "t3"
        outline_format = "truetype"
    elif outline_mode == "truetype_segment":
        model_name = "cls"
        outline_format = "truetype"
    else:
        model_name = "cls"
        outline_format = "postscript"

    model = StyleClassifierLM(
        model=model_name,
        num_layers=3,
        emb_size=64,
        nhead=4,
        class_labels=class_labels,
        dim_feedforward=128,
        dropout=0.1,
        lr=0.01,
        warmup_steps=256,
        outline_format=outline_format,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="font-classifier-{epoch:02d}-{val_loss:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=512,
        devices="auto",
        accelerator="auto",
        precision="16-mixed",
        callbacks=[checkpoint_callback, lr_monitor],
    )

    trainer.fit(model, datamodule=data_module)

    trainer.test(ckpt_path="best", datamodule=data_module)
