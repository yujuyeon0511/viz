import torch
import torch.nn as nn

from transformers import SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig


class SIGLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        # cos_sim_fc = nn.CosineSimilarity(dim=-1)
        cls = image_features[:, 0].unsqueeze(1)
        #image_features_fuse = image_forward_outs.hidden_states[-3][:, 1:]
        if self.select_feature == 'patch':
            image_features = image_features#[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        """
        """
        patch_features = image_features
        # return image_features
        bsz = image_features.shape[0]
        l = int(image_features.shape[1] ** 0.5)
        dim = image_features.shape[2]
        image_features = image_features.reshape(bsz, l, l, dim)
        #image_features_fuse = image_features_fuse.reshape(bsz, l, l, dim)
        # 336
        a = '336'
        if a == '336':
            center_x = int(l / 2)
            center_y = int(l / 2)
            min_x = center_x - 1
            max_x = center_x + 1
            min_y = center_y - 1
            max_y = center_y + 1

            out = []
            out_fuse = []
            for i in range(0, int(l / 2), 2):
               i_x = min_x - i -1
               j_x = max_x + i +1
               i_y = min_y - i -1
               j_y = max_y + i +1
               out.append(image_features[:, i_x:j_x, i_y:j_y, :])
               #out_fuse.append(image_features_fuse[:, i_x:j_x, i_y:j_y, :])

        # 224
        elif a == '224':
            center_x = int(l / 2)
            center_y = int(l / 2)
            min_x = center_x - 1
            max_x = center_x + 1
            min_y = center_y - 1
            max_y = center_y + 1

            out = []
            for i in range(int(l / 2)):
               i_x = min_x - i
               j_x = max_x + i
               i_y = min_y - i
               j_y = max_y + i
               out.append(image_features[:, i_x:j_x, i_y:j_y, :])
        # cls
        #out.append(cls)
        return out, patch_features#, out_fuse

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs)#.to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


