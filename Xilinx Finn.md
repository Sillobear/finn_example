## Installation: 
- [ ] Install Docker to run without root
- [ ] Set up FINN_XILINX_PATH and FINN_XILINX_VERSION environment Example: ```FINN_XILINX_PATH=/tools/Xilinx``` and ```FINN_XILINX_VERSION=2024.1```
- [ ] Clone the FINN Compiler: ```git clone https://github.com/Xilinx/finn/``` and ```cd finn``` in to the directory
## Notebook: 
### Run Notebook Server
- [ ] Execute ```bash ./run-docker.sh notebook``` to launch a Docker container with a jupyter notebook server running
When using PyCharm:
- [ ] Copy the last of the three shown URLs (e.g. ```http://127.0.0.1:8888/tree?token=a982b7290e6eaaf3e2db4499718d67d1085debfd14b8ce29```)
- [ ] Go to ```Tools > Add Jupyter Connection > Connect to Jupyter Server using URL``` and paste the copied URL
You can now execute every Notebook in the notebooks folder

### Shutdown notebook server:
**Attention:** Check which process is your jupyter process by `ps aux | grep jupyter`
- [ ] ```CTRL + C```
- [ ] `docker exec -it <name_of_container> bash` you can do `docker ps`to get the name of your container
- [ ] Kill the jupyter-notebook process (e.g. `kill 7`)

# Usage: 
## Tips:
- There are very good Notebooks in the `notebooks` folder for every step 
- Also there are two End-To-End Examples, the `bnn-pynq`Example only makes sense if you have a PYNQ board
## General Steps:
1. Train a QNN with Brevitas
2. Export that QNN to QONNX and convert to FINN-ONNX
3. Use `build_dataflow` to get a stitched ip, estimates or something like this
## Train a QNN with Brevitas:
https://xilinx.github.io/brevitas/getting_started
Model: 
```
import brevitas  
import brevitas.nn as qnn  
from brevitas.export import export_qonnx  
from brevitas.quant import Int8Bias  
  
class QuantNeuralNet(nn.Module):  
    def __init__(self):  
        super(QuantNeuralNet, self).__init__()  
        self.flatten = nn.Flatten()  
        self.quant = qnn.QuantIdentity(bit_width=bits, return_quant_tensor=True)  
        self.linear1 = qnn.QuantLinear(in_features=28*28, out_features=512, bias=True, weight_bit_width=bits, bias_quant=Int8Bias)  
        self.relu1 = qnn.QuantReLU(bit_width=bits, return_quant_tensor=True)  
        self.linear2 = qnn.QuantLinear(in_features=512, out_features=512, bias=True, weigth_bit_width=bits, bias_quant=Int8Bias)  
        self.relu2 = qnn.QuantReLU(bit_width=bits, return_quant_tensor=True)  
        self.linear3 = qnn.QuantLinear(in_features=512, out_features=10, bias=True, weight_bit_width=bits, bias_quant=Int8Bias)  
  
    def forward(self, x):  
        x = self.flatten(x)  
        x = self.quant(x)  
        out = self.linear1(x)  
        out = self.relu1(out)  
        out = self.linear2(out)  
        out = self.relu2(out)  
        logits = self.linear3(out)  
        return logits  
  
model = QuantNeuralNet()

# Train the Model
```
## Export to QONNX:
```
export_qonnx_path = "mnist_model_q.onnx"  
input_shape = (1, 28, 28)  
export_qonnx(model, torch.randn(input_shape), export_qonnx_path)
```
## Clean Up the ONNX Model
```
from qonnx.util.cleanup import cleanup  
  
export_onnx_path_cleaned = "mnist_model_q_clean.onnx"  
cleanup(export_qonnx_path, out_file=export_onnx_path_cleaned)
```
## Import QONNX Model into FINN
```
from qonnx.core.modelwrapper import ModelWrapper

model = ModelWrapper(export_onnx_path_cleaned)
```
## Convert to FINN by using QONNXtoFINN
```
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN  
from qonnx.core.datatype import DataType  
  
#model.set_tensor_datatype(model.graph.input[0].name), DataType[""])  
model = model.transform(ConvertQONNXtoFINN())  
  
export_onnx_path_converted = "mnist_model_q_converted.onnx"  
model.save(export_onnx_path_converted)
```
## Builds
### Estimate Reports:
```
import finn.builder.build_dataflow as build  
import finn.builder.build_dataflow_config as build_cfg  
import os  
import shutil

model_file = "mnist_model_q_converted.onnx"  
  
estimates_output_dir = "output_estimates_only"  
  
  
#Delete previous run results if exist  
if os.path.exists(estimates_output_dir):  
    shutil.rmtree(estimates_output_dir)  
    print("Previous run results deleted!")  
  
  
cfg_estimates = build.DataflowBuildConfig(  
    output_dir          = estimates_output_dir,  
    mvau_wwidth_max     = 80,  
    target_fps          = 1000000,  
    synth_clk_period_ns = 10.0,  
    fpga_part           = "xc7a35ticsg324-1L",  
    steps               = build_cfg.estimate_only_dataflow_steps,  
    generate_outputs=[  
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,  
    ]  
)

build.build_dataflow_cfg(model_file, cfg_estimates)
```
## Stitched IP:
Actual output Artefact suited for our use case
```
import finn.builder.build_dataflow as build  
import finn.builder.build_dataflow_config as build_cfg  
import os  
import shutil  
  
model_file = "mnist_model_q_converted.onnx"  
  
rtlsim_output_dir = "output_ipstitch_rtlsim"  
  
#Delete previous run results if exist  
if os.path.exists(rtlsim_output_dir):  
    shutil.rmtree(rtlsim_output_dir)  
    print("Previous run results deleted!")  
  
cfg_stitched_ip = build.DataflowBuildConfig(  
    output_dir          = rtlsim_output_dir,  
    mvau_wwidth_max     = 80,  
    target_fps          = 1000000,  
    synth_clk_period_ns = 10.0,  
    fpga_part           = "xc7a35ticsg324-1L",  
    generate_outputs=[  
        build_cfg.DataflowOutputType.STITCHED_IP,  
    ]  
)

build.build_dataflow_cfg(model_file, cfg_stitched_ip)
```

I started FINN Docker with `bash ./run-docker.sh`and used the shell to execute stitched_ip.py seen above. 
Output:
```
AssertionError: IPGen failed: /tmp/finn_dev_silas/code_gen_ipgen_MVAU_hls_1_7pj3i_u5/project_MVAU_hls_1/sol1/impl/ip not found. Check log under /tmp/finn_dev_silas/code_gen_ipgen_MVAU_hls_1_7pj3i_u5
```
This is the Log: 
![[vitis_hls.log]]
We get 5 errors looking like this one:
```ERROR: [HLS 207-2163] 'bitwidth' attribute requires integer constant between 1 and 8191 inclusive (/tools/Xilinx/Vitis_HLS/2024.1/common/technology/autopilot/etc/ap_common.h:521:29)```

## Toy Example:
Input: Binary 15 Bit
Output: Binary 5 Bit