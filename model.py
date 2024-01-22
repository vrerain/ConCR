# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


class ContextTree():
    def __init__(self, value):
        self.value = value
        self.children = []
        self.parent = None

    def add(self, child):
        self.children.append(child)
    
    def setParent(self, parent):
        self.parent = parent

def createTree(packages, funcs, levels, route=None, up_fun_num=None):
    root = ContextTree("Code_Context")
    for pack in packages:
        package_node = ContextTree(pack.reshape(1, -1))
        package_node.setParent(root)
        root.add(package_node)
    curLevel = -1
    curNode = root
    for i in range(len(funcs)):
        level = levels[i]
        if (level == -1):
            break
        if up_fun_num is not None and i >= up_fun_num:
            break
        f = funcs[i]
        f = f.reshape(1, -1)
        if route is not None and route[i] != 1:
            continue
        if level > curLevel and level - curLevel == 1:
            child = ContextTree(f)
            child.setParent(curNode)
            curNode.add(child)
            curNode = child
            curLevel = level
            continue
        if level == curLevel:
            child = ContextTree(f)
            child.setParent(curNode.parent)
            curNode.parent.add(child)
            curNode = child
            continue
        if level < curLevel:
            nums = curLevel - level
            child = ContextTree(f)
            p = curNode.parent
            for i in range(nums):
                p = p.parent
            child.setParent(p)
            p.add(child)
            curLevel = level
            curNode = child
            continue
    return root

def Han(inputs, query_emb):
    inputs = torch.unsqueeze(inputs, 0)
    u = torch.tanh(inputs)  #[]
    att = torch.matmul(u, query_emb.reshape(-1, 1))
    att_score = F.softmax(att, dim=1)
    scored_inputs = inputs * att_score

    return torch.sum(scored_inputs, dim=1)

    
class Model(nn.Module):   
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.HanMlp = nn.Linear(768, 768)
        self.FuMlp = nn.Linear(768, 1)
        
    def forward(self, code_inputs=None, attn_mask=None,position_idx=None, nl_inputs=None, context_inputs=None, levels=None, routes=None, up_fun_num=None): 
        if code_inputs is not None:
            nodes_mask=position_idx.eq(0)
            token_mask=position_idx.ge(2)        
            inputs_embeddings=self.encoder.embeddings.word_embeddings(code_inputs)
            nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
            nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
            avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
            inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
            return self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[1]
        else:
            old_nl_vec = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[1]  #[batch_size, hidden_dim]
            if levels is None:
                return old_nl_vec

            package_inputs = context_inputs[0]
            function_inputs = context_inputs[1]
            temp_context_embs = []
            for i in range(package_inputs.shape[0]):
                self.query_emb = old_nl_vec[i]
                package_input = package_inputs[i]
                function_input = function_inputs[i]

                with torch.no_grad():
                    package_input = self.encoder(package_input, attention_mask=package_input.ne(1))[1]
                    function_input = self.encoder(function_input, attention_mask=function_input.ne(1))[1]

                level = levels[i]
                if routes is not None:
                    route = routes[i]
                    temp_route_embs = []
                    for r in route:
                        root2 = createTree(package_input, function_input, level, route=r)
                        route_emb = self.getContextPre(root2)
                        temp_route_embs.append(route_emb)
                    temp_context_embs.append(Han(torch.concat(temp_route_embs, 0), self.query_emb))
                else:
                    root = createTree(package_input, function_input, level, up_fun_num=up_fun_num[i])
                    route_emb = self.getContextPre(root)
                    temp_context_embs.append(torch.concat([route_emb], 0))
            context_embs = torch.concat(temp_context_embs, 0)
            fusion_vec = torch.concat([old_nl_vec.unsqueeze(1), context_embs.unsqueeze(1)], dim=1)
            return self.FusionAttention(fusion_vec)
        
            
    def FusionAttention(self, inputs):
        a = torch.softmax(self.FuMlp(inputs), 1)
        return (inputs * a).sum(dim=1).squeeze(1)

    def getContextPre(self, root):
        if (len(root.children) == 0) :
            return self.HanMlp(root.value)
        else:
            res = []
            for child in root.children:
                res.append(self.getContextPre(child))
            Child_emb = Han(torch.concat(res, 0), self.query_emb)
            if (root.value == "Code_Context"):
                return Child_emb
            Cur_emb = self.HanMlp(root.value)
            return Han(torch.concat([Cur_emb, Child_emb], 0), self.query_emb)
      
        
 
