import torch
import torch.nn as nn
import torchvision.ops as ops

from lib.Correlation_Volume import Correalation
from lib.Encoder_Decoder import Backbone
from transformers import Owlv2TextModel

from .ops import ConvBNReLU, resize_to

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class CTC(nn.Module):
    def __init__(self, in_planes, pyramid_type='conv'):
        super(CTC, self).__init__()
        self.pyramid_type = pyramid_type
        self.nl_layer1 = Correalation(in_planes*2, None, False, scale=2)
        self.nl_layer2 = Correalation(in_planes*2, None, False, scale=4)
        self.nl_layer3 = Correalation(in_planes*2, None, False, scale=8)

        self.conv1 = BasicConv2d(in_planes*2, in_planes, 3, padding=1)
        self.conv2 = BasicConv2d(in_planes*2, in_planes, 3, padding=1)
        self.conv3 = BasicConv2d(in_planes*2, in_planes, 3, padding=1)

    def forward(self, fea):
        #pdb.set_trace()
        out = []
        for i in range(1, len(fea)):
            input1 = torch.cat([fea[0][0], fea[i][0]], dim=1)
            input2 = torch.cat([fea[0][1], fea[i][1]], dim=1)
            input3 = torch.cat([fea[0][2], fea[i][2]], dim=1)
            res1 = self.conv1(self.nl_layer1(input1))
            res2 = self.conv2(self.nl_layer2(input2))
            res3 = self.conv3(self.nl_layer3(input3))
            out.append([res1, res2, res3])
        return out

class FSP(nn.Module):
    def __init__(self, in_c, num_groups=6, hidden_dim=None, num_frames=1, **kwargs):
        super(FSP, self).__init__()
        self.num_groups = num_groups
        self.attention_type = "temporal_only"
        self.num_frames = num_frames

        hidden_dim = hidden_dim or in_c // 2
        expand_dim = hidden_dim * num_groups
        self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_groups * hidden_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, num_groups * hidden_dim, 1),
            nn.Softmax(dim=1),
        )

        self.interact = nn.ModuleDict()
        self.interact["0"] = ConvBNReLU(hidden_dim, 3 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 3 * hidden_dim, 3, 1, 1)
        self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)

        self.BNReLU = ConvBNReLU(num_groups * hidden_dim, in_c, 3, 1, 1,act_name=None)
        self.final_relu = nn.ReLU(True)

    def forward(self, x):
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1)
        outs = []
        gates = []

        group_id = 0
        curr_x = xs[group_id]
        branch_out = self.interact[str(group_id)](curr_x)
        curr_out, curr_fork, curr_gate = branch_out.chunk(3, dim=1)
        outs.append(curr_out)
        gates.append(curr_gate)

        for group_id in range(1, self.num_groups - 1):
            curr_x = torch.cat([xs[group_id], curr_fork], dim=1)
            branch_out = self.interact[str(group_id)](curr_x)
            curr_out, curr_fork, curr_gate = branch_out.chunk(3, dim=1)
            outs.append(curr_out)
            gates.append(curr_gate)

        group_id = self.num_groups - 1
        curr_x = torch.cat([xs[group_id], curr_fork], dim=1)
        branch_out = self.interact[str(group_id)](curr_x)
        curr_out, curr_gate = branch_out.chunk(2, dim=1)
        outs.append(curr_out)
        gates.append(curr_gate)

        out = torch.cat(outs, dim=1)
        gate = self.gate_genator(torch.cat(gates, dim=1))
        out = out*gate
        out = self.BNReLU(out)
        out = self.final_relu(out + x)
        return  out 

class ImageModel(nn.Module):

    def __init__(self, args):
        super(ImageModel, self).__init__()
        self.args = args
        self.backbone = Network(pvtv2_pretrained=self.args.pvtv2_pretrained, imgsize=self.args.trainsize)
    def forward(self, frame):
        seg = self.backbone(frame)
        return seg

class VLJGS(nn.Module):
    def __init__(self, args):
        super(VLJGS, self).__init__()

        self.args = args
        self.extra_channels = 0
        self.text_length = len(args.text)

        self.backbone = Backbone(pvtv2_pretrained=False, imgsize=self.args.trainsize)
        if self.args.pretrained_cod10k is not None:
            self.load_backbone(self.args.pretrained_cod10k)
            
        self.nlpmodel = Owlv2TextModel.from_pretrained("google/owlv2-base-patch16")
        self.visual_proj = nn.Linear(32, 512,bias=False)
        self.fmap_proj = nn.Linear(self.text_length,32,bias=False)

        self.ctc = CTC(in_planes=32, pyramid_type='conv')
        self.fsp1 = FSP(in_c=32, hum_groups=6, num_frames=1)
        self.fsp2 = FSP(in_c=32, hum_groups=6, num_frames=1)
        self.fsp3 = FSP(in_c=32, hum_groups=6, num_frames=1)

        self.fusion_conv = nn.Sequential(nn.Conv2d(2, 32, 3, 1, 1),
                                         nn.Conv2d(32, 32, 3, 1, 1),
                                         nn.Conv2d(32, 1, 3, 1, 1),
            )

    def load_backbone(self, pretrained):
        pretrained_dict = torch.load(pretrained)
        model_dict = self.state_dict()
        print("Load pretrained cod10k parameters from {}".format(pretrained))

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def freeze_bn(self):
        for m in self.backbone.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eval()
                
    def VLF(self, fmap_ori, text):
        fmap_after_merge = []
        for i in range(len(fmap_ori)):
            fmap = [self.visual_proj(fmap_ori[i][0].permute(0,2,3,1)), self.visual_proj(fmap_ori[i][1].permute(0,2,3,1)), self.visual_proj(fmap_ori[i][2].permute(0,2,3,1))]
            merge_fmap = [torch.matmul(fmap[0], text), torch.matmul(fmap[1], text), torch.matmul(fmap[2], text)]
            conversion_fmap = [self.fmap_proj(merge_fmap[0]).permute(0,3,1,2), self.fmap_proj(merge_fmap[1]).permute(0,3,1,2), self.fmap_proj(merge_fmap[2]).permute(0,3,1,2)]
            fmap_after_merge.append(conversion_fmap)

        return fmap_after_merge
    
    def forward(self, image, text=None):
        fmap_ori = []
        for i in range(len(image)):
            fmap_ori += [self.backbone.encoder(image[i])] #[3,3,4,32,44,44]
        
        text_output = self.nlpmodel(**text)
        pooled_output = text_output.pooler_output
        pooled_output = pooled_output.unsqueeze(0).unsqueeze(0).permute(0,1,3,2) #[1,1,512, language length]
        fmap_before_cab = self.VLF(fmap_ori, pooled_output) 
      
        corr_vol = self.ctc(fmap_before_cab)  #[2,3,4,32,44,44]  
        after_fsp = [None, None, None]

        out_prime = []
        for i in range(0, len(corr_vol)):
            after_fsp[0] = self.fsp1(fmap_before_cab[0][0]+ corr_vol[i][0])
            after_fsp[1] = self.fsp2(fmap_before_cab[0][1]+ corr_vol[i][1])
            after_fsp[2] = self.fsp3(fmap_before_cab[0][2]+ corr_vol[i][2])
            decoder_out = self.backbone.decoder(after_fsp)
            out_prime += [decoder_out]       
        concated = torch.cat(out_prime, dim=1)
        res = self.fusion_conv(concated)
        res_frame12 = out_prime[0]
        res_frame13 = out_prime[1]

        return  res_frame12, res_frame13, res 
