import torch
import safetensors

MODEL_TYPE=["7B", "30B"]

model_7b_0_path = "/share/work/sh/work/20241016/InternEvo/ckpt/0/model_tp0_pp0.pt"
model_7b_0_path = "/share/work/sh/work/20241016/InternEvo/ckpt02/0/model_tp0_pp0.pt"

path10="/share/work/sh/work/20241016/weight/Chameleon_7B_mGPT/model-00001-of-00002.safetensors"
path11="/share/work/sh/work/20241016/weight/Chameleon_7B_mGPT/model-00002-of-00002.safetensors"
path20="/share/work/sh/work/20241016/InternEvo/ckpt/0_hf/model-00001-of-00002.safetensors"
path21="/share/work/sh/work/20241016/InternEvo/ckpt/0_hf/model-00002-of-00002.safetensors"

def show_weight(path):
    model = torch.load(model_7b_0_path, map_location="cpu")

    for i in model.keys():
        tensor = model[i]
        print(f"key:{i}, shape:{tensor.shape}, dtype:{tensor.dtype}")

# show_weight(model_7b_0_path)

def compare_weight(path1, path2):
    f1 = []
    f1_keys = []
    f2 = []
    f2_keys = []
    for path in path1:        
        f1.append(safetensors.safe_open(path, framework="pt", device='cpu'))
        f1_keys.append(f1[-1].keys())
    for path in path2:        
        f2.append(safetensors.safe_open(path, framework="pt", device='cpu'))
        f2_keys.append(f2[-1].keys())
    # f1_keys=f1.keys()
    # f2_keys=f2.keys()
    print(f"1 KEYS:{f1_keys}")
    print(f"2 KEYS:{f2_keys}")

    print(f1_keys == f2_keys)

    for id2,f2_key in enumerate(f2_keys):
        for key in f2_key:
            not_in = True
            for id1,f1_key in enumerate(f1_keys):
                if key not in f1_key:
                    continue
                else:
                    tensor2 = f2[id2].get_tensor(key)
                    tensor1 = f1[id1].get_tensor(key)
                    print(f"key:{key}, equal:{torch.equal(tensor2, tensor1)}")
                    not_in = False
                    break
            if not_in:
                print(key)

compare_weight([path10,path11], [path20,path21])


    # import pdb;pdb.set_trace()
    # ckpt_list=f.keys()
    # for i in ckpt_list:
    #     # tensor_slice = f.get_slice(i)
    #     tensor_slice = f.get_tensor(i)
    #     # print(f"key:{i}, shape:{tensor_slice.get_shape()}")
    #     print(f"key:{i}, shape:{tensor_slice.shape},dtype:{tensor_slice.dtype}")
    #     # print(f"value:{f.get_tensor(i)}")
    #     del tensor_slice

