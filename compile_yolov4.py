import tvm
from tvm import relay
import numpy as np
from tvm.contrib.download import download_testdata
import torch
from tvm.contrib import graph_runtime
from PIL import Image
from torchvision import transforms
from models import Yolov4
model = Yolov4(inference=True)
model.eval()
input_shape = [1, 3, 512, 512]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

img_url = "https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((512, 512))

my_preprocess = transforms.Compose(
    [
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
img = my_preprocess(img)
img = np.expand_dims(img, 0)

input_name = "input0"
shape_list = [(input_name, img.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
print(f"Mod", mod)

target = "llvm"
target_host = "llvm"
ctx = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, target_host=target_host, params=params)

lib.export_library("yolov4_tvm.so")

dtype = "float32"
m = graph_runtime.GraphModule(lib["default"](ctx))
# Set inputs
m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
# Execute
m.run()
# Get outputs
tvm_output = m.get_output(0).asnumpy()

# Assert Correctness
#
model_output = model(torch.Tensor(img.astype(dtype)))[0].detach().numpy()
np.testing.assert_allclose(model_output, tvm_output, rtol=1e-05, atol=1e-05)
#
