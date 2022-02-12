import torch
import numpy as np


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


class VanillaGradient(SaliencyMask):
    def __init__(self, model):
        super(VanillaGradient, self).__init__(model)

    def get_mask(self, image_tensor, target_class=None):
        image_tensor = image_tensor.clone()
        image_tensor.requires_grad = True
        image_tensor.retain_grad()

        logits,_,_ = self.model(image_tensor)
        target = torch.zeros_like(logits)
        target[0][target_class if target_class else logits.topk(1, dim=1)[1]] = 1
        self.model.zero_grad()
        logits.backward(target)


        return np.moveaxis(image_tensor.grad.detach().cpu().numpy()[0], 0, -1)

    def get_smoothed_mask(self, image_tensor, target_class=None, samples=25, std=0.15, process=lambda x: x**2):
        std = std * (torch.max(image_tensor) - torch.min(image_tensor)).detach().cpu().numpy()

        batch, channels, width, height = image_tensor.size()
        grad_sum = np.zeros((width, height, channels))
        for sample in range(samples):
            noise = torch.empty(image_tensor.size()).normal_(0, std).to(image_tensor.device)
            noise_image = image_tensor + noise
            grad_sum += process(self.get_mask(noise_image, target_class))
        return grad_sum / samples

    @staticmethod
    def apply_region(mask, region):
        return mask * region[..., np.newaxis]



class IntegratedGradients(VanillaGradient):
    def get_mask(self, image_tensor, target_class=None, baseline='black', steps=25, process=lambda x: x):
        if baseline is 'black':
            baseline = torch.ones_like(image_tensor) * torch.min(image_tensor).detach().cpu()
        elif baseline is 'white':
            baseline = torch.ones_like(image_tensor) * torch.max(image_tensor).detach().cpu()
        else:
            baseline = torch.zeros_like(image_tensor)

        batch, channels, place = image_tensor.size()
        grad_sum = np.zeros((place, channels))
        image_diff = image_tensor - baseline

        for step, alpha in enumerate(np.linspace(0, 1, steps)):
            print('Processing Integrated Gradients at literation: ', step)
            image_step = baseline + alpha * image_diff
            grad_sum += process(super(IntegratedGradients, self).get_mask(image_step, target_class))
        return grad_sum * np.moveaxis(image_diff.detach().cpu().numpy()[0], 0, -1) / steps

if __name__ == "__main__":
    IG = IntegratedGradients()


