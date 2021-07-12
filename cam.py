from pytorch_gram_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def constructcam(model,target_layer,cuda): 
    cam = GradCAM(model=model,target_layer=target_layer,use_cuda=cuda)

    cam()