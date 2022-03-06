import torch
import numpy as np
import cv2

class SaliencyMask(object):
    def __init__(self, model):
        if torch.cuda.is_available() == True:
            self.model = model.cuda()
        else:
            self.model = model
        self.model.eval()
        self.gradient = None
        self.hooks = list()

    def get_mask(self, image_tensor, target_class=None):
        raise NotImplementedError('A derived class should implemented this method')

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


class VanillaGradient():
    def __init__(self, helpers):
        self.helpers = helpers


    def get_mask(self, image_tensor, target_class=None, box=None):
        image_tensor = image_tensor.clone()
        image_tensor.requires_grad = True
        image_tensor.retain_grad()

        loss = 0
        for helper in self.helpers:
            if box is None:
                al, _ = helper.attack_loss(image_tensor)
            else:
                al, _, _ = helper.loss_in_box(image_tensor, box=box)
            loss += al
        if loss==0:
            return np.zeros(image_tensor.shape)

        loss.backward()
        return image_tensor.grad.detach().cpu().numpy()

    def get_smoothed_mask(self,
            image_tensor, 
            target_class=None,
            samples=25,
            std=0.15,
            process=lambda x:x):

        std = std * (torch.max(image_tensor) - torch.min(image_tensor)).detach().cpu().numpy()

        batch, channels, width, height = image_tensor.size()
        grad_sum = np.zeros((width, height, channels))
        for sample in range(samples):
            noise = torch.empty(image_tensor.size()).normal_(0, std).to(image_tensor.device)
            noise_image = image_tensor + noise
            grad_sum += process(self.get_mask(noise_image, target_class))
        return grad_sum / samples

    def get_grad_mask(self, image_tensor, samples=15, std=5, process=lambda x: x**2, baseline=None, box=None):
        grad_sum = np.zeros(image_tensor.shape)
        for sample in range(samples):
            noise = torch.empty(image_tensor.size()).normal_(0, std).to(image_tensor.device)
            noise_image = image_tensor + noise
            grad_sum += process(super(IntegratedGradients, self).get_mask(noise_image, box=box))
        return grad_sum.sum(-1)


    @staticmethod
    def apply_region(mask, region):
        return mask * region[..., np.newaxis]



class IntegratedGradients(VanillaGradient):
    def get_mask(self, image_tensor, target_class=None, baseline='black', steps=10, process=lambda x: x, box=None, attack_type='integrated_grad'):

        if attack_type=="grad":
            grad = super(IntegratedGradients, self).get_mask(image_tensor, target_class, box=box)
            return grad.sum(-1)
        elif attack_type=="grad_input":
            grad = super(IntegratedGradients, self).get_mask(image_tensor, target_class, box=box)
            return (grad * image_tensor.detach().cpu().numpy()).sum(-1)
        elif attack_type=='random':
            return np.random.uniform(size=image_tensor.shape[:2])
        elif attack_type=="integrated_grad":
            H, W, C = image_tensor.size()
            grad_sum = np.zeros((H,W,C))
            image_diff = image_tensor - baseline

            #mask = super(IntegratedGradients, self).get_mask(image_tensor, box=box)
            #return mask.sum(-1)


            for step, alpha in enumerate(np.linspace(0, 1, steps)):
                #print('Processing Integrated Gradients at literation: ', step)
                image_step = baseline + alpha * image_diff
                grad_sum += process(super(IntegratedGradients, self).get_mask(image_step, target_class, box=box))
            return (grad_sum * image_diff.detach().cpu().numpy() / steps).sum(-1)

if __name__ == "__main__":
    from faster_helper import Helper as FasterHelper
    helper = FasterHelper()
    IG = IntegratedGradients([helper])

    img_path = "./images/836.png"
    img = cv2.imread(img_path)
    img = torch.from_numpy(img).float()

    baseline = torch.ones_like(img) * torch.min(img).detach().cpu()
    mask = IG.get_mask(img, baseline=baseline)

    mask = np.uint8( mask / mask.max() * 255 )
    mask =  cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    img = np.uint8( img.numpy() * 0.8 + mask * 0.2 )
    cv2.imwrite("demo.png", img)






