#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from .multimodal_projector.builder import build_gen_img_projector
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
# from diffusers.utils.torch_utils import randn_tensor
import wandb

cos_loss_fn = nn.CosineSimilarity()


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

        version = 'pooling'
        if version == 'pooling':
            a = 'conv336_pooling_sfe'
        elif version == 'cropping':
            a = 'conv336_cropping_sfe_dfi'

        if a == 'conv336':
            self.conv_linear = nn.Sequential(nn.GELU(),
                                             nn.Linear(512, 4096, bias=False))
            self.conv_4 = nn.Conv2d(1024, 512, 4, bias=False)
            self.conv_8 = nn.Conv2d(1024, 512, 8, bias=False)
            self.conv_12 = nn.Conv2d(1024, 512, 12, bias=False)
            self.conv_16 = nn.Conv2d(1024, 512, 16, bias=False)
            self.conv_20 = nn.Conv2d(1024, 512, 20, bias=False)
            self.conv_24 = nn.Conv2d(1024, 512, 24, bias=False)

        if a == 'conv336_3_tokens':
            self.conv_linear = nn.Sequential(nn.GELU(),
                                             nn.Linear(512, 4096, bias=False))
            self.conv_8 = nn.Conv2d(1024, 512, 8, bias=False)
            self.conv_16 = nn.Conv2d(1024, 512, 16, bias=False)
            self.conv_24 = nn.Conv2d(1024, 512, 24, bias=False)

        elif a == 'conv336_12_tokens':
            self.conv_linear = nn.Sequential(nn.GELU(),
                                             nn.Linear(512, 4096, bias=False))
            self.conv_2 = nn.Conv2d(1024, 512, 2, bias=False)
            self.conv_4 = nn.Conv2d(1024, 512, 4, bias=False)
            self.conv_6 = nn.Conv2d(1024, 512, 6, bias=False)
            self.conv_8 = nn.Conv2d(1024, 512, 8, bias=False)
            self.conv_10 = nn.Conv2d(1024, 512, 10, bias=False)
            self.conv_12 = nn.Conv2d(1024, 512, 12, bias=False)
            self.conv_14 = nn.Conv2d(1024, 512, 14, bias=False)
            self.conv_16 = nn.Conv2d(1024, 512, 16, bias=False)
            self.conv_18 = nn.Conv2d(1024, 512, 18, bias=False)
            self.conv_20 = nn.Conv2d(1024, 512, 20, bias=False)
            self.conv_22 = nn.Conv2d(1024, 512, 22, bias=False)
            self.conv_24 = nn.Conv2d(1024, 512, 24, bias=False)


        elif a == 'conv336_conv_49':

            self.conv_linear = nn.Sequential(nn.GELU(),
                                             nn.Linear(512, 4096, bias=False))

            self.conv_CM = nn.Conv2d(1024, 512, 12, stride=2, bias=False)


        elif a == 'conv336_pooling_sfe':
            self.conv_linear = nn.Sequential(nn.GELU(),
                                             nn.Linear(512, 4096, bias=False))

            self.conv_4 = nn.Sequential(nn.AdaptiveAvgPool2d(4),
                                        nn.Conv2d(1024, 512, 4, bias=False))

            self.conv_8 = nn.Sequential(nn.AdaptiveAvgPool2d(8),
                                        nn.Conv2d(1024, 512, 8, bias=False))

            self.conv_12 = nn.Sequential(nn.AdaptiveAvgPool2d(12),
                                         nn.Conv2d(1024, 512, 12, bias=False))

            self.conv_16 = nn.Sequential(nn.AdaptiveAvgPool2d(16),
                                         nn.Conv2d(1024, 512, 16, bias=False))

            self.conv_20 = nn.Sequential(nn.AdaptiveAvgPool2d(20),
                                         nn.Conv2d(1024, 512, 20, bias=False))

            self.conv_24 = nn.Conv2d(1024, 512, 24, bias=False)


        elif a == 'conv336_adptive_pooling_multifuse_q':
            self.conv_linear = nn.Sequential(nn.GELU(),
                                             nn.Linear(1024, 4096))

            self.conv_4 = nn.Sequential(nn.AdaptiveAvgPool2d(4),
                                        nn.Conv2d(1024, 512, 4, bias=False))

            self.conv_8 = nn.Sequential(nn.AdaptiveAvgPool2d(8),
                                        nn.Conv2d(1024, 512, 8, bias=False))

            self.conv_12 = nn.Sequential(nn.AdaptiveAvgPool2d(12),
                                         nn.Conv2d(1024, 512, 12, bias=False))

            self.conv_16 = nn.Sequential(nn.AdaptiveAvgPool2d(16),
                                         nn.Conv2d(1024, 512, 16, bias=False))

            self.conv_20 = nn.Sequential(nn.AdaptiveAvgPool2d(20),
                                         nn.Conv2d(1024, 512, 20, bias=False))

            self.conv_24 = nn.Conv2d(1024, 512, 24, bias=False)

            self.query_projector_small = nn.Sequential(nn.LayerNorm(512),
                                                       nn.Linear(512, 512))
            self.key_projector_small = nn.Sequential(nn.LayerNorm(512),
                                                     nn.Linear(512, 512))
            self.value_projector_small = nn.Sequential(nn.LayerNorm(512),
                                                       nn.Linear(512, 512))
            self.conv_small = nn.Conv2d(1024, 512, 12, stride=2, bias=False)

            self.query_projector_mid = nn.Sequential(nn.LayerNorm(512),
                                                     nn.Linear(512, 512))
            self.key_projector_mid = nn.Sequential(nn.LayerNorm(512),
                                                   nn.Linear(512, 512))
            self.value_projector_mid = nn.Sequential(nn.LayerNorm(512),
                                                     nn.Linear(512, 512))
            self.conv_mid = nn.Conv2d(1024, 512, 9, stride=2, bias=False)

            self.query_projector_huge = nn.Sequential(nn.LayerNorm(512),
                                                      nn.Linear(512, 512))
            self.key_projector_huge = nn.Sequential(nn.LayerNorm(512),
                                                    nn.Linear(512, 512))
            self.value_projector_huge = nn.Sequential(nn.LayerNorm(512),
                                                      nn.Linear(512, 512))
            self.conv_huge = nn.Conv2d(1024, 512, 5, stride=2, bias=False)

            self.query_projector_small = nn.Sequential(nn.LayerNorm(512),
                                                       nn.Linear(512, 512))
            self.key_projector_small = nn.Sequential(nn.LayerNorm(512),
                                                     nn.Linear(512, 512))
            self.value_projector_small = nn.Sequential(nn.LayerNorm(512),
                                                       nn.Linear(512, 512))
            self.conv_small = nn.Conv2d(1024, 512, 12, stride=2, bias=False)




        elif a == 'conv336_crop_multifuse_q':
            self.conv_linear = nn.Sequential(nn.GELU(),
                                             nn.Linear(2048, 4096, bias=False)
                                             )

            self.query_projector_small = nn.Sequential(nn.LayerNorm(512),
                                                       nn.Linear(512, 512))
            self.key_projector_small = nn.Sequential(nn.LayerNorm(512),
                                                     nn.Linear(512, 512))
            self.value_projector_small = nn.Sequential(nn.LayerNorm(512),
                                                       nn.Linear(512, 512))
            self.conv_small = nn.Conv2d(1024, 512, 12, stride=2, bias=False)

            self.query_projector_mid = nn.Sequential(nn.LayerNorm(512),
                                                     nn.Linear(512, 512))
            self.key_projector_mid = nn.Sequential(nn.LayerNorm(512),
                                                   nn.Linear(512, 512))
            self.value_projector_mid = nn.Sequential(nn.LayerNorm(512),
                                                     nn.Linear(512, 512))
            self.conv_mid = nn.Conv2d(1024, 512, 9, stride=2, bias=False)

            self.query_projector_huge = nn.Sequential(nn.LayerNorm(512),
                                                      nn.Linear(512, 512))
            self.key_projector_huge = nn.Sequential(nn.LayerNorm(512),
                                                    nn.Linear(512, 512))
            self.value_projector_huge = nn.Sequential(nn.LayerNorm(512),
                                                      nn.Linear(512, 512))
            self.conv_huge = nn.Conv2d(1024, 512, 5, stride=2, bias=False)

            self.query_projector_small = nn.Sequential(nn.LayerNorm(512),
                                                       nn.Linear(512, 512))
            self.key_projector_small = nn.Sequential(nn.LayerNorm(512),
                                                     nn.Linear(512, 512))
            self.value_projector_small = nn.Sequential(nn.LayerNorm(512),
                                                       nn.Linear(512, 512))
            self.conv_small = nn.Conv2d(1024, 512, 12, stride=2, bias=False)

            self.conv_4 = nn.Conv2d(1024, 512, 4, bias=False)
            self.conv_8 = nn.Conv2d(1024, 512, 8, bias=False)
            self.conv_12 = nn.Conv2d(1024, 512, 12, bias=False)
            self.conv_16 = nn.Conv2d(1024, 512, 16, bias=False)
            self.conv_20 = nn.Conv2d(1024, 512, 20, bias=False)
            self.conv_24 = nn.Conv2d(1024, 512, 24, bias=False)



        elif a == 'tfm336_adptive_pooling':

            self.tfm_linear = nn.Sequential(nn.GELU(),
                                            nn.Linear(1024, 4096, bias=False))

            self.tfm_4 = nn.Transformer(batch_first=True, norm_first=True,
                                        d_model=1024, num_encoder_layers=4, num_decoder_layers=4,
                                        dim_feedforward=1024, dropout=0.1, nhead=8, bias=False)
            self.tfm_8 = nn.Transformer(batch_first=True, norm_first=True,
                                        d_model=1024, num_encoder_layers=4, num_decoder_layers=4,
                                        dim_feedforward=1024, dropout=0.1, nhead=8, bias=False)
            self.tfm_12 = nn.Transformer(batch_first=True, norm_first=True,
                                         d_model=1024, num_encoder_layers=4, num_decoder_layers=4,
                                         dim_feedforward=1024, dropout=0.1, nhead=8, bias=False)
            self.tfm_16 = nn.Transformer(batch_first=True, norm_first=True,
                                         d_model=1024, num_encoder_layers=4, num_decoder_layers=4,
                                         dim_feedforward=1024, dropout=0.1, nhead=8, bias=False)
            self.tfm_20 = nn.Transformer(batch_first=True, norm_first=True,
                                         d_model=1024, num_encoder_layers=4, num_decoder_layers=4,
                                         dim_feedforward=1024, dropout=0.1, nhead=8, bias=False)
            self.tfm_24 = nn.Transformer(batch_first=True, norm_first=True,
                                         d_model=1024, num_encoder_layers=4, num_decoder_layers=4,
                                         dim_feedforward=1024, dropout=0.1, nhead=8, bias=False)

            self.query_4 = nn.Parameter(torch.zeros(1, 1, 1024))
            self.query_8 = nn.Parameter(torch.zeros(1, 1, 1024))
            self.query_12 = nn.Parameter(torch.zeros(1, 1, 1024))
            self.query_16 = nn.Parameter(torch.zeros(1, 1, 1024))
            self.query_20 = nn.Parameter(torch.zeros(1, 1, 1024))
            self.query_24 = nn.Parameter(torch.zeros(1, 1, 1024))


        elif a == 'conv336_adptive_pooling_fuse_49':
            self.conv_linear = nn.Sequential(nn.GELU(),
                                             nn.Linear(1024, 4096, bias=False))

            self.query_projector = nn.Sequential(nn.LayerNorm(512),
                                                 nn.Linear(512, 512))
            self.key_projector = nn.Sequential(nn.LayerNorm(512),
                                               nn.Linear(512, 512))
            self.value_projector = nn.Sequential(nn.LayerNorm(512),
                                                 nn.Linear(512, 512))

            self.conv_4 = nn.Sequential(nn.AdaptiveAvgPool2d(4),
                                        nn.Conv2d(1024, 512, 4, bias=False))

            self.conv_8 = nn.Sequential(nn.AdaptiveAvgPool2d(8),
                                        nn.Conv2d(1024, 512, 8, bias=False))

            self.conv_12 = nn.Sequential(nn.AdaptiveAvgPool2d(12),
                                         nn.Conv2d(1024, 512, 12, bias=False))

            self.conv_16 = nn.Sequential(nn.AdaptiveAvgPool2d(16),
                                         nn.Conv2d(1024, 512, 16, bias=False))

            self.conv_20 = nn.Sequential(nn.AdaptiveAvgPool2d(20),
                                         nn.Conv2d(1024, 512, 20, bias=False))

            self.conv_24 = nn.Conv2d(1024, 512, 24, bias=False)

            self.conv_CM = nn.Conv2d(1024, 512, 16, stride=2, bias=False)

        elif a == 'conv336_cropping_sfe_dfi':
            self.conv_linear = nn.Sequential(nn.GELU(),
                                             nn.Linear(1024, 4096))

            self.query_projector = nn.Sequential(nn.LayerNorm(512),
                                                 nn.Linear(512, 512))
            self.key_projector = nn.Sequential(nn.LayerNorm(512),
                                               nn.Linear(512, 512))
            self.value_projector = nn.Sequential(nn.LayerNorm(512),
                                                 nn.Linear(512, 512))

            self.conv_4 = nn.Conv2d(1024, 512, 4, bias=False)
            self.conv_8 = nn.Conv2d(1024, 512, 8, bias=False)
            self.conv_12 = nn.Conv2d(1024, 512, 12, bias=False)
            self.conv_16 = nn.Conv2d(1024, 512, 16, bias=False)
            self.conv_20 = nn.Conv2d(1024, 512, 20, bias=False)
            self.conv_24 = nn.Conv2d(1024, 512, 24, bias=False)

            self.conv_CM = nn.Conv2d(1024, 512, 16, stride=2, bias=False)


        elif a == 'conv336_CM':
            self.conv_linear = nn.Sequential(nn.GELU(),
                                             nn.Linear(1024, 4096, bias=False))

            self.conv_4 = nn.Conv2d(1024, 1024, 4, bias=False)
            self.conv_8 = nn.Conv2d(1024, 1024, 8, bias=False)
            self.conv_12 = nn.Conv2d(1024, 1024, 12, bias=False)
            self.conv_16 = nn.Conv2d(1024, 1024, 16, bias=False)
            self.conv_20 = nn.Conv2d(1024, 1024, 20, bias=False)
            self.conv_24 = nn.Conv2d(1024, 1024, 24, bias=False)

            self.conv_CM = nn.Conv2d(1024, 1024, 12, stride=2, bias=False)
            encoder_layer = nn.TransformerDecoderLayer(batch_first=True, norm_first=True,
                                                       d_model=1024,
                                                       dim_feedforward=1024, dropout=0.1, nhead=8, bias=False)
            # self.CM_P = nn.Transformer(batch_first=True, norm_first=True,
            #                d_model=1024, num_encoder_layers=2, num_decoder_layers=2,
            #                dim_feedforward=1024, dropout=0.1, nhead=8, bias=False)
            decoder_norm = nn.LayerNorm(1024)
            # # # self.CM_L = nn.TransformerDecoder(encoder_layer, num_layers=1, norm=decoder_norm)
            self.CM_P = nn.TransformerDecoder(encoder_layer, num_layers=2, norm=decoder_norm)
            # for p in self.CM_P.parameters():
            #     nn.init.constant_(p, 0)
            # for p in self.CM_L.parameters():
            #     nn.init.constant_(p, 0)

            # self.gate_L = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            # self.gate_P = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            # self.gate_L.data.fill_(0.4)
            # self.gate_P.data.fill_(0.5)
        elif a == 'conv336_24':
            self.conv_linear = nn.Sequential(nn.GELU(),
                                             nn.Linear(512, 4096, bias=False))
            self.conv_24 = nn.Conv2d(1024, 512, 24, bias=False)



        elif a == 'conv224':
            self.conv_linear = nn.Sequential(nn.GELU(),
                                             nn.Linear(512, 4096, bias=False))
            self.conv_1 = nn.Conv2d(1024, 512, 2, bias=False)
            self.conv_2 = nn.Conv2d(1024, 512, 4, bias=False)
            self.conv_3 = nn.Conv2d(1024, 512, 6, bias=False)
            self.conv_4 = nn.Conv2d(1024, 512, 8, bias=False)
            self.conv_5 = nn.Conv2d(1024, 512, 10, bias=False)
            self.conv_6 = nn.Conv2d(1024, 512, 12, bias=False)
            self.conv_7 = nn.Conv2d(1024, 512, 14, bias=False)
            self.conv_8 = nn.Conv2d(1024, 512, 16, bias=False)


        elif a == 'tfm336':

            self.tfm_linear = nn.Sequential(  # nn.Linear(1024, 1024),
                nn.GELU(),
                nn.Linear(1024, 4096, bias=False))

            self.tfm_4 = nn.Transformer(batch_first=True, norm_first=True,
                                        d_model=1024, num_encoder_layers=4, num_decoder_layers=4,
                                        dim_feedforward=1024, dropout=0.1, nhead=8, bias=False)
            self.tfm_8 = nn.Transformer(batch_first=True, norm_first=True,
                                        d_model=1024, num_encoder_layers=4, num_decoder_layers=4,
                                        dim_feedforward=1024, dropout=0.1, nhead=8, bias=False)
            self.tfm_12 = nn.Transformer(batch_first=True, norm_first=True,
                                         d_model=1024, num_encoder_layers=4, num_decoder_layers=4,
                                         dim_feedforward=1024, dropout=0.1, nhead=8, bias=False)
            self.tfm_16 = nn.Transformer(batch_first=True, norm_first=True,
                                         d_model=1024, num_encoder_layers=4, num_decoder_layers=4,
                                         dim_feedforward=1024, dropout=0.1, nhead=8, bias=False)
            self.tfm_20 = nn.Transformer(batch_first=True, norm_first=True,
                                         d_model=1024, num_encoder_layers=4, num_decoder_layers=4,
                                         dim_feedforward=1024, dropout=0.1, nhead=8, bias=False)
            self.tfm_24 = nn.Transformer(batch_first=True, norm_first=True,
                                         d_model=1024, num_encoder_layers=4, num_decoder_layers=4,
                                         dim_feedforward=1024, dropout=0.1, nhead=8, bias=False)

            self.query_4 = nn.Parameter(torch.zeros(1, 1, 1024))
            self.query_8 = nn.Parameter(torch.zeros(1, 1, 1024))
            self.query_12 = nn.Parameter(torch.zeros(1, 1, 1024))
            self.query_16 = nn.Parameter(torch.zeros(1, 1, 1024))
            self.query_20 = nn.Parameter(torch.zeros(1, 1, 1024))
            self.query_24 = nn.Parameter(torch.zeros(1, 1, 1024))


        elif a == 'conv_block':
            self.c = nn.Conv2d(1024, 512, 3, bias=False),
            self.conv_linear = nn.Sequential(nn.GELU(),
                                             nn.Linear(512, 4096, bias=False))

            self.conv_4 = nn.Sequential(

                # nn.Conv2d(1024, 512, 4, bias=False),
                nn.Conv2d(1024, 512, 3, bias=False),
                nn.BatchNorm2d(512),
                # nn.GELU(),
                nn.Conv2d(512, 512, 2, bias=False),
                nn.BatchNorm2d(512),
            )

            self.conv_8 = nn.Sequential(
                nn.Conv2d(1024, 512, 5, bias=False),
                nn.BatchNorm2d(512),
                # nn.GELU(),
                nn.Conv2d(512, 512, 3, bias=False),
                nn.BatchNorm2d(512),
                # nn.GELU(),
                nn.Conv2d(512, 512, 2, bias=False),
                nn.BatchNorm2d(512),
            )

            self.conv_12 = nn.Sequential(
                nn.Conv2d(1024, 1024, 5, bias=False),
                nn.BatchNorm2d(1024),
                # nn.GELU(),
                nn.Conv2d(1024, 512, 3, bias=False),
                nn.BatchNorm2d(512),
                # nn.GELU(),
                nn.Conv2d(512, 512, 5, bias=False),
                nn.BatchNorm2d(512),
                # nn.GELU(),
                nn.Conv2d(512, 512, 2, bias=False),
                nn.BatchNorm2d(512),
            )

            self.conv_16 = nn.Sequential(
                nn.Conv2d(1024, 1024, 5, bias=False),
                nn.BatchNorm2d(1024),
                # nn.GELU(),
                nn.Conv2d(1024, 1024, 3, bias=False),
                nn.BatchNorm2d(1024),
                # nn.GELU(),
                nn.Conv2d(1024, 512, 5, bias=False),
                nn.BatchNorm2d(512),
                # nn.GELU(),
                nn.Conv2d(512, 512, 3, bias=False),
                nn.BatchNorm2d(512),
                # nn.GELU(),
                nn.Conv2d(512, 512, 3, bias=False),
                nn.BatchNorm2d(512),
                # nn.GELU(),
                nn.Conv2d(512, 512, 2, bias=False),
                nn.BatchNorm2d(512),
            )

            self.conv_20 = nn.Sequential(
                nn.Conv2d(1024, 1024, 5, bias=False),
                nn.BatchNorm2d(1024),
                # nn.GELU(),
                nn.Conv2d(1024, 1024, 5, bias=False),
                nn.BatchNorm2d(1024),
                # nn.GELU(),
                nn.Conv2d(1024, 1024, 3, bias=False),
                nn.BatchNorm2d(1024),
                # nn.GELU(),
                nn.Conv2d(1024, 512, 5, bias=False),
                nn.BatchNorm2d(512),
                # nn.GELU(),
                nn.Conv2d(512, 512, 3, bias=False),
                nn.BatchNorm2d(512),
                # nn.GELU(),
                nn.Conv2d(512, 512, 3, bias=False),
                nn.BatchNorm2d(512),
                # nn.GELU(),
                nn.Conv2d(512, 512, 2, bias=False),
                nn.BatchNorm2d(512),
            )

            self.conv_24 = nn.Sequential(
                nn.Conv2d(1024, 1024, 5, bias=False),
                nn.BatchNorm2d(1024),
                # nn.GELU(),
                nn.Conv2d(1024, 1024, 5, bias=False),
                nn.BatchNorm2d(1024),
                # nn.GELU(),
                nn.Conv2d(1024, 1024, 5, bias=False),
                nn.BatchNorm2d(1024),
                # nn.GELU(),
                nn.Conv2d(1024, 512, 3, bias=False),
                nn.BatchNorm2d(512),
                # nn.GELU(),
                nn.Conv2d(512, 512, 5, bias=False),
                nn.BatchNorm2d(512),
                # nn.GELU(),
                nn.Conv2d(512, 512, 3, bias=False),
                nn.BatchNorm2d(512),
                # nn.GELU(),
                nn.Conv2d(512, 512, 3, bias=False),
                nn.BatchNorm2d(512),
                # nn.GELU(),
                nn.Conv2d(512, 512, 2, bias=False),
                nn.BatchNorm2d(512),
            )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_gen_modules(self, model_args):

        self.config.text_emb_to_img_layers = model_args.text_emb_to_img_layers
        self.config.llm_num_hidden_layers = model_args.llm_num_hidden_layers
        self.config.llm_hidden_size = model_args.llm_hidden_size
        self.config.nums_gen_img_tokens = model_args.nums_gen_img_tokens
        self.config.num_clip_tokens = model_args.num_clip_tokens
        self.config.text_fc_to_img_mode = model_args.text_fc_to_img_mode
        self.gen_img_projector = build_gen_img_projector(model_args)

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        local_features, patch_features = image_features
        res = []

        version = 'pooling'

        if version == 'pooling':
            a = 'conv336_pooling_sfe'
        elif version == 'cropping':
            a = 'conv336_cropping_sfe_dfi'


        # # 336
        if a == 'conv336' or a == 'conv_block':
            p = self.get_model().mm_projector(patch_features)
            res.append(self.get_model().conv_4(local_features[0].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_8(local_features[1].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_12(local_features[2].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_16(local_features[3].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_20(local_features[4].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_24(local_features[5].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            c = self.get_model().conv_linear(torch.cat(res, dim=1))
            image_features = torch.cat([c, p], dim=1)

        if a == 'conv336_3_tokens':
            p = self.get_model().mm_projector(patch_features)
            res.append(self.get_model().conv_8(local_features[0].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_16(local_features[1].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))

            res.append(self.get_model().conv_24(local_features[2].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            c = self.get_model().conv_linear(torch.cat(res, dim=1))
            image_features = torch.cat([c, p], dim=1)


        elif a == 'conv336_12_tokens':
            p = self.get_model().mm_projector(patch_features)
            res.append(self.get_model().conv_2(local_features[0].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_4(local_features[1].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_6(local_features[2].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_8(local_features[3].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_10(local_features[4].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_12(local_features[5].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_14(local_features[6].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_16(local_features[7].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_18(local_features[8].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_20(local_features[9].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_22(local_features[10].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_24(local_features[11].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            c = self.get_model().conv_linear(torch.cat(res, dim=1))
            image_features = torch.cat([c, p], dim=1)











        elif a == 'conv336_CM':
            # p = self.get_model().mm_projector(patch_features)
            res.append(self.get_model().conv_4(local_features[0].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_8(local_features[1].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_12(local_features[2].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_16(local_features[3].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_20(local_features[4].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_24(local_features[5].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))

            c = torch.cat(res, dim=1).nan_to_num()
            conv_fuse = self.get_model().conv_CM(local_features[5].permute(0, 3, 1, 2)).reshape(c.shape[0], c.shape[2],
                                                                                                -1).permute(0, 2,
                                                                                                            1).nan_to_num()

            # P fuse
            # cc = self.get_model().CM_L(c, patch_features)

            # embed_query = self.get_model().query_projector(c)
            # embed_key = self.get_model().key_projector(conv_fuse)
            # embed_value = self.get_model().val_projector(conv_fuse)
            # embed_att = embed_query[:, :, None] @ (embed_key.transpose(-1, -2) / (embed_key.shape[-1] ** 0.5))
            # embed_att = embed_att.nan_to_num()
            # embed_feat = (embed_att.softmax(-1) @ embed_value).mean(2)

            pp = self.get_model().CM_P(patch_features, conv_fuse).nan_to_num()
            # P = pp + patch_features
            P = torch.cat([patch_features, pp], dim=-1)
            # wandb.log({'gate c': self.get_model().gate_L.data.detach().float()})
            # wandb.log({'gate p': self.get_model().gate_P.data.detach().float()})
            image_features = torch.cat(
                [self.get_model().conv_linear(c), self.get_model().mm_projector(P)], dim=1).nan_to_num()

        elif a == 'conv336_cropping_sfe_dfi':
            # p = self.get_model().mm_projector(patch_features)
            res.append(self.get_model().conv_4(local_features[0].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_8(local_features[1].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_12(local_features[2].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_16(local_features[3].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_20(local_features[4].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_24(local_features[5].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))

            c = torch.cat(res, dim=1).nan_to_num()
            conv_fuse = self.get_model().conv_CM(local_features[5].permute(0, 3, 1, 2)).reshape(c.shape[0], c.shape[2],
                                                                                                -1).permute(0, 2,
                                                                                                            1).nan_to_num()

            embed_query = self.get_model().query_projector(c)
            embed_key = self.get_model().key_projector(conv_fuse)
            embed_value = self.get_model().value_projector(conv_fuse)
            embed_att = embed_query @ (embed_key.transpose(-1, -2) / (embed_key.shape[-1] ** 0.5))
            embed_att = embed_att.nan_to_num()

            # torch.save(embed_att, 'crop_49_blue.pt')
            embed_feat = (embed_att.softmax(-1) @ embed_value)
            embed_fuse = torch.cat([c, embed_feat], dim=-1)
            image_features = torch.cat(
                [self.get_model().conv_linear(embed_fuse), self.get_model().mm_projector(patch_features)], dim=1)

        elif a == 'conv336_adptive_pooling_fuse_49':
            # p = self.get_model().mm_projector(patch_features)
            res.append(self.get_model().conv_4(local_features.permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_8(local_features.permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_12(local_features.permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_16(local_features.permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_20(local_features.permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_24(local_features.permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))

            c = torch.cat(res, dim=1).nan_to_num()
            conv_fuse = self.get_model().conv_CM(local_features.permute(0, 3, 1, 2)).reshape(c.shape[0], c.shape[2],
                                                                                             -1).permute(0, 2,
                                                                                                         1).nan_to_num()
            embed_query = self.get_model().query_projector(c)
            embed_key = self.get_model().key_projector(conv_fuse)
            embed_value = self.get_model().value_projector(conv_fuse)
            embed_att = embed_query @ (embed_key.transpose(-1, -2) / (embed_key.shape[-1] ** 0.5))
            embed_att = embed_att.nan_to_num()

            # torch.save(embed_att, 'pooling_49_clis_1-1.pt')

            embed_feat = (embed_att.softmax(-1) @ embed_value)
            embed_fuse = torch.cat([c, embed_feat], dim=-1)
            image_features = torch.cat(
                [self.get_model().conv_linear(embed_fuse), self.get_model().mm_projector(patch_features)], dim=1)

        elif a == 'conv336_crop_multifuse_q':
            res.append(self.get_model().conv_4(local_features[0].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_8(local_features[1].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_12(local_features[2].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_16(local_features[3].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_20(local_features[4].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_24(local_features[5].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))

            c = torch.cat(res, dim=1).nan_to_num()
            # small
            c_small = self.get_model().conv_small(local_features[5].permute(0, 3, 1, 2)).reshape(c.shape[0], c.shape[2],
                                                                                                 -1).permute(0, 2,
                                                                                                             1).nan_to_num()
            embed_query = self.get_model().query_projector_small(c)
            embed_key = self.get_model().key_projector_small(c_small)
            embed_value = self.get_model().value_projector_small(c_small)
            embed_att = embed_query @ (embed_key.transpose(-1, -2) / (embed_key.shape[-1] ** 0.5))
            embed_att = embed_att.nan_to_num()

            embed_fuse_small = (embed_att.softmax(-1) @ embed_value)

            # mid
            c_mid = self.get_model().conv_mid(local_features[5].permute(0, 3, 1, 2)).reshape(c.shape[0], c.shape[2],
                                                                                             -1).permute(0, 2,
                                                                                                         1).nan_to_num()
            embed_query = self.get_model().query_projector_mid(c)
            embed_key = self.get_model().key_projector_mid(c_mid)
            embed_value = self.get_model().value_projector_small(c_mid)
            embed_att = embed_query @ (embed_key.transpose(-1, -2) / (embed_key.shape[-1] ** 0.5))
            embed_att = embed_att.nan_to_num()

            embed_fuse_mid = (embed_att.softmax(-1) @ embed_value)

            # huge
            c_huge = self.get_model().conv_huge(local_features[5].permute(0, 3, 1, 2)).reshape(c.shape[0], c.shape[2],
                                                                                               -1).permute(0, 2,
                                                                                                           1).nan_to_num()
            embed_query = self.get_model().query_projector_huge(c)
            embed_key = self.get_model().key_projector_huge(c_huge)
            embed_value = self.get_model().value_projector_huge(c_huge)
            embed_att = embed_query @ (embed_key.transpose(-1, -2) / (embed_key.shape[-1] ** 0.5))
            embed_att = embed_att.nan_to_num()

            embed_fuse_huge = (embed_att.softmax(-1) @ embed_value)

            # embed_fuse = embed_fuse_small + embed_fuse_mid + embed_fuse_huge
            embed_fuse = torch.cat([c, embed_fuse_small, embed_fuse_mid, embed_fuse_huge], dim=-1)

            image_features = torch.cat(
                [self.get_model().conv_linear(embed_fuse), self.get_model().mm_projector(patch_features)], dim=1)

        elif a == 'conv336_adptive_pooling_multifuse_q':
            res.append(self.get_model().conv_4(local_features[5].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_8(local_features[5].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_12(local_features[5].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_16(local_features[5].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_20(local_features[5].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_24(local_features[5].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))

            c = torch.cat(res, dim=1)
            # small
            c_small = self.get_model().conv_small(local_features[5].permute(0, 3, 1, 2)).reshape(c.shape[0], c.shape[2],
                                                                                                 -1).permute(0, 2, 1)
            embed_query = self.get_model().query_projector_small(c)
            embed_key = self.get_model().key_projector_small(c_small)
            embed_value = self.get_model().value_projector_small(c_small)
            embed_att = embed_query @ (embed_key.transpose(-1, -2) / (embed_key.shape[-1] ** 0.5))
            embed_att = embed_att.nan_to_num()

            embed_fuse_small = (embed_att.softmax(-1) @ embed_value)

            # mid
            c_mid = self.get_model().conv_mid(local_features[5].permute(0, 3, 1, 2)).reshape(c.shape[0], c.shape[2],
                                                                                             -1).permute(0, 2, 1)
            embed_query = self.get_model().query_projector_mid(c)
            embed_key = self.get_model().key_projector_mid(c_mid)
            embed_value = self.get_model().value_projector_small(c_mid)
            embed_att = embed_query @ (embed_key.transpose(-1, -2) / (embed_key.shape[-1] ** 0.5))
            embed_att = embed_att.nan_to_num()

            embed_fuse_mid = (embed_att.softmax(-1) @ embed_value)

            # huge
            c_huge = self.get_model().conv_huge(local_features[5].permute(0, 3, 1, 2)).reshape(c.shape[0], c.shape[2],
                                                                                               -1).permute(0, 2, 1)
            embed_query = self.get_model().query_projector_huge(c)
            embed_key = self.get_model().key_projector_huge(c_huge)
            embed_value = self.get_model().value_projector_huge(c_huge)
            embed_att = embed_query @ (embed_key.transpose(-1, -2) / (embed_key.shape[-1] ** 0.5))
            embed_att = embed_att.nan_to_num()

            embed_fuse_huge = (embed_att.softmax(-1) @ embed_value)

            embed_fuse = embed_fuse_small + embed_fuse_mid + embed_fuse_huge
            embed_fuse = torch.cat([c, embed_fuse], dim=-1)

            image_features = torch.cat(
                [self.get_model().conv_linear(embed_fuse), self.get_model().mm_projector(patch_features)], dim=1)


        elif a == 'conv336_conv_49':

            conv_49 = self.get_model().conv_CM(local_features[5].permute(0, 3, 1, 2)).reshape(patch_features.shape[0],
                                                                                              512, -1).permute(0, 2, 1)

            image_features = torch.cat(
                [self.get_model().conv_linear(conv_49), self.get_model().mm_projector(patch_features)], dim=1)

        elif a == 'conv336_pooling_sfe':
            p = self.get_model().mm_projector(patch_features)

            res.append(self.get_model().conv_4(local_features.permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_8(local_features.permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_12(local_features.permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_16(local_features.permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_20(local_features.permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_24(local_features.permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))

            c = self.get_model().conv_linear(torch.cat(res, dim=1))
            image_features = torch.cat([c, p], dim=1)


        elif a == 'conv336_24':
            p = self.get_model().mm_projector(patch_features)
            res.append(self.get_model().conv_24(local_features[5].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            c = self.get_model().conv_linear(torch.cat(res, dim=1))
            image_features = torch.cat([c, p], dim=1)

        # 224
        elif a == 'conv224':
            p = self.get_model().mm_projector(patch_features)
            res.append(self.get_model().conv_1(local_features[0].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_2(local_features[1].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_3(local_features[2].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_4(local_features[3].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_5(local_features[4].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_6(local_features[5].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_7(local_features[6].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            res.append(self.get_model().conv_8(local_features[7].permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
            c = self.get_model().conv_linear(torch.cat(res, dim=1))
            image_features = torch.cat([c, p], dim=1)

        elif a == 'tfm336_adptive_pooling':
            p = self.get_model().mm_projector(patch_features)

            res.append(self.get_model().tfm_4(
                nn.functional.adaptive_avg_pool2d(local_features[5].permute(0, 3, 1, 2), 4).reshape(
                    local_features[0].shape[0], -1, local_features[0].shape[3])
                , self.get_model().query_4.repeat(local_features[0].shape[0], 1, 1)))
            res.append(self.get_model().tfm_8(
                nn.functional.adaptive_avg_pool2d(local_features[5].permute(0, 3, 1, 2), 8).reshape(
                    local_features[0].shape[0], -1, local_features[0].shape[3])
                , self.get_model().query_8.repeat(local_features[0].shape[0], 1, 1)))
            res.append(self.get_model().tfm_12(
                nn.functional.adaptive_avg_pool2d(local_features[5].permute(0, 3, 1, 2), 12).reshape(
                    local_features[0].shape[0], -1, local_features[0].shape[3])
                , self.get_model().query_12.repeat(local_features[0].shape[0], 1, 1)))
            res.append(self.get_model().tfm_16(
                nn.functional.adaptive_avg_pool2d(local_features[5].permute(0, 3, 1, 2), 16).reshape(
                    local_features[0].shape[0], -1, local_features[0].shape[3])
                , self.get_model().query_16.repeat(local_features[0].shape[0], 1, 1)))
            res.append(self.get_model().tfm_20(
                nn.functional.adaptive_avg_pool2d(local_features[5].permute(0, 3, 1, 2), 20).reshape(
                    local_features[0].shape[0], -1, local_features[0].shape[3])
                , self.get_model().query_20.repeat(local_features[0].shape[0], 1, 1)))
            res.append(self.get_model().tfm_24(
                nn.functional.adaptive_avg_pool2d(local_features[5].permute(0, 3, 1, 2), 24).reshape(
                    local_features[0].shape[0], -1, local_features[0].shape[3])
                , self.get_model().query_24.repeat(local_features[0].shape[0], 1, 1)))

            c = self.get_model().tfm_linear(torch.cat(res, dim=1))
            image_features = torch.cat([c, p], dim=1)

        elif a == 'tfm336':
            p = self.get_model().mm_projector(patch_features)

            res.append(self.get_model().tfm_4(
                local_features[0].reshape(local_features[0].shape[0], -1, local_features[0].shape[3])
                , self.get_model().query_4.repeat(local_features[0].shape[0], 1, 1)))
            res.append(self.get_model().tfm_8(
                local_features[1].reshape(local_features[0].shape[0], -1, local_features[0].shape[3])
                , self.get_model().query_8.repeat(local_features[0].shape[0], 1, 1)))
            res.append(self.get_model().tfm_12(
                local_features[2].reshape(local_features[0].shape[0], -1, local_features[0].shape[3])
                , self.get_model().query_12.repeat(local_features[0].shape[0], 1, 1)))
            res.append(self.get_model().tfm_16(
                local_features[3].reshape(local_features[0].shape[0], -1, local_features[0].shape[3])
                , self.get_model().query_16.repeat(local_features[0].shape[0], 1, 1)))
            res.append(self.get_model().tfm_20(
                local_features[4].reshape(local_features[0].shape[0], -1, local_features[0].shape[3])
                , self.get_model().query_20.repeat(local_features[0].shape[0], 1, 1)))
            res.append(self.get_model().tfm_24(
                local_features[5].reshape(local_features[0].shape[0], -1, local_features[0].shape[3])
                , self.get_model().query_24.repeat(local_features[0].shape[0], 1, 1)))

            c = self.get_model().tfm_linear(torch.cat(res, dim=1))
            image_features = torch.cat([c, p], dim=1)

        return image_features

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels,
            images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)  # [6=bxn,3,336,336]
            image_features = self.encode_images(concat_images)  # [6,576+6,5120]
            split_sizes = [image.shape[0] for image in images]  # [3,3]
            image_features = torch.split(image_features, split_sizes, dim=0)  # ([3,582,5120], [3,582,5120])
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        sp_image_feature = image_feature[:, 0:6, :]
                        image_feature = image_feature[:, 6:, :]
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx],
                                                                                            self.config.image_grid_pinpoints,
                                                                                            self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(
                                    image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    # llava - 1.6 + SP
                    image_feature = torch.cat((sp_image_feature.flatten(0, 1), image_feature), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)
        # image_features = image_features.to(images.device.index)
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device,
                                   dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype,
                                       device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                              device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                             device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def cal_acc(self, output, labels):
        chosen_tokens = torch.max(output.logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
        labels = labels[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)  # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)
        return gen_acc

    def l2_loss(self, u, v):
        """
        Args:
          u: (N, T_I_V_A.txt, D) tensor.
          v: (N, T_I_V_A.txt, D) tensor.
        Returns:
          l1_loss: (N,) tensor of summed L1 loss.
        """
        assert u.shape == v.shape, (u.shape, v.shape)
        return ((u - v) ** 2).sum(dim=-1) ** 0.5

    def cos_loss(self, u, v):
        return 1 - cos_loss_fn(u, v)

        return

    def learn_noise(self, prompt_embed, label):
        noise_pred = self.get_model().pipeline.unet(
            self.get_model().random_noise,
            24,
            encoder_hidden_states=prompt_embed,
            timestep_cond=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
        )[0]
        mse_loss = self.l2_loss(noise_pred, label).to(self.device)
        mse_loss = mse_loss.mean()
        return mse_loss

    def prepare_align_stablediffusion(self, target, output, align_signal, prompt_embed):
        hidden_states = []
        # text_hidden_fcs = self.gen_text_hidden_fcs
        # based on the targets to obtain the hidden state, targets includes the [BOS] token
        """only cal t2i data"""
        align_target = []
        align_hidden = []
        align_signal_emb = []
        new_target = []
        t = torch.stack(output.hidden_states)
        for i, it in enumerate(target):
            if (it == self.gen_img_token_idx[0]).sum() != torch.tensor(0):
                align_target.append(it)
                align_hidden.append(t[:, i, :, :])
                align_signal_emb.append(align_signal[i])
                new_target.append(target[i])
        align_target = torch.stack(align_target)
        align_hidden = torch.stack(align_hidden).transpose(0, 1)
        align_text_emb = torch.stack(align_signal_emb)
        new_target = torch.stack(new_target)

        start_pos = (new_target == self.gen_img_token_idx[0]).nonzero(as_tuple=False)[:, 1].tolist()
        end_pos = (new_target == self.gen_img_token_idx[-1]).nonzero(as_tuple=False)[:, 1].tolist()
        # logging.info(f'targets : {targets}')
        # logging.info(f'start_pos : {start_pos}')
        # logging.info(f'end_pos : {end_pos}')
        assert 0 < len(start_pos) == len(end_pos) == align_target.size(0) and len(end_pos) > 0, (start_pos, end_pos)
        num_gen_tokens = len(self.gen_img_token_idx)
        for idx, fc_layer in zip([-1], self.model.gen_img_projector):
            hidden_embedding = []
            input_embedding = []
            hidden_states = []
            for b, (s, e) in enumerate(zip(start_pos, end_pos)):
                assert e - s + 1 == num_gen_tokens, (s, e)
                hidden_embedding.append(align_hidden[idx][b, s:e + 1, :])
                input_embedding.append(self.model.get_input_embeddings()(new_target[b, s:e + 1]))
            hidden_embedding = torch.stack(hidden_embedding, dim=0)
            input_embedding = torch.stack(input_embedding, dim=0)
            hidden_states.append(fc_layer(hidden_embedding, input_embedding))  # (N, seq_len, 2048)
        embeddings = torch.stack(hidden_states, dim=-1).sum(dim=-1).reshape(-1, 4, 64, 64)  # (N, 77, 768)

        # embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # (N, T_I_V_A.txt, 256)

        mse_loss = self.cos_loss(embeddings.reshape(embeddings.shape[0], -1),
                                 align_text_emb.reshape(embeddings.shape[0], -1)).to(self.device)
        mse_loss = mse_loss.mean()

        return mse_loss

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
