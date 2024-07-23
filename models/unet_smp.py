import segmentation_models_pytorch as smp

def unetsmp(in_channels, classes):
    return smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes,
    )
