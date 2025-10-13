import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import functional as f
#import torch.utils.checkpoint as cp

class MaxPoolMatMulLayer(nn.Module):

      def __init__(self, kernel_size=3, stride=1, padding=1):

          super(MaxPoolMatMulLayer, self).__init__()
          self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
          #torch.set_default_dtype(torch.float16)

      def forward(self, x, initial_input):

          # Apply MaxPool2D
          pooled_output = self.pool(x)
          matmul_result = pooled_output*initial_input

          # To combine with spatial dimensions, we might need to reshape or recombine
          return matmul_result

class MaxPoolMatMulStack(nn.Module):

      def __init__(self, num_layers=128, kernel_size=3, stride=1, padding=1):

          super(MaxPoolMatMulStack, self).__init__()
          self.num_layers = num_layers
          module_list = [MaxPoolMatMulLayer(kernel_size, stride, padding) for _ in range(num_layers)]
          self.layers = nn.ModuleList(module_list)
          #torch.set_default_dtype(torch.float16)
          

      def forward(self, x):

          initial_input = x.clone()
          current_tensor = -x

          current_tensor[:, :, [0, -1], :] = 1
          current_tensor[:, :, :, [0, -1]] = 1

          # Apply the sequence of MaxPool and MatMul layers
          for layer in self.layers:
              current_tensor = layer(current_tensor, initial_input)

          current_tensor0 = F.normalize(current_tensor)
          current_tensor1 = current_tensor0 * initial_input

          return current_tensor1


class PhysicsFormer(nn.Module):

      def __init__(self,invalid_pair_list=None):

          super(PhysicsFormer,self).__init__()
          self.T=128
          self.invalid_pair_list = invalid_pair_list
          self.maxpooling = MaxPoolMatMulStack(num_layers=self.T, kernel_size=3, stride=1, padding=1)
          self.relu = nn.ReLU(inplace=False)
          self.empty_norm = torch.tensor([0,],dtype=torch.float16)
          
      def final_operation(self, original,mode='opening'):

          if mode == 'opening':

             opened = self.maxpooling(original)

             #     torch.save(opened,f'/cluster/work/cvl/shbasu/opened_{idx}.pt')
             subtracted = original- opened
             #torch.save(subtracted,f'/cluster/work/cvl/shbasu/subtracted_{idx}.pt')       

          l1_norm = torch.norm(subtracted, p=1)
          return l1_norm

      def forward(self, input):
                    
          final_norm = 0
          pairs = []

          logits_upscaled = f.interpolate(input,size=(256,256),mode='bilinear',align_corners=False)
          
          for idx, pair in enumerate(self.invalid_pair_list):

              concatenated_tensor = torch.cat((logits_upscaled[:,pair[0]:pair[0]+1,::], logits_upscaled[:,pair[1]:pair[1]+1,::]), dim=1)
              softmax = torch.softmax(concatenated_tensor,dim=1)
              #torch.save(softmax,f'/cluster/work/cvl/shbasu/softmax_{idx}.pt')
              difference = softmax[:, 0:1, :, :] - softmax[:, 1:2, :, :]
              relu = self.relu(difference)
              #torch.save(relu,f'/cluster/work/cvl/shbasu/relu_{idx}.pt')
              pairs.append(relu)

          final_concatenated = torch.cat(pairs, dim=1)
          del pairs
          norm_opened_1 = self.final_operation(final_concatenated)
          if torch.any(norm_opened_1.isnan()) or norm_opened_1.numel()==0:
              final_norm = self.empty_norm.cuda()             
          else:
              norm_opened_1 = torch.clamp(norm_opened_1,max=65504)
              final_norm = norm_opened_1.half()
              
          return final_norm
