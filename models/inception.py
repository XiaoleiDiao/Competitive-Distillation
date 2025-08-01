import torchvision

def inceptionV1(num_classes=100):
    model = torchvision.models.GoogLeNet(num_classes)
    return model
