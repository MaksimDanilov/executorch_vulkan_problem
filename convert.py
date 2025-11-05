import os
import torch

from pathlib import Path
from model import U2NET

root_dirpath = os.path.dirname(os.path.realpath(__file__))

class Frontend(U2NET):
    def __init__(self):
        super().__init__(3, 1)
        self.load_state_dict(
            torch.load(os.path.join(root_dirpath, 'saved_models', 'u2net', 'u2net.pth'), 
                       weights_only=True,
                       map_location=torch.device('cpu')))
        self.eval()

    def preprocess(self, image: torch.Tensor):
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=image.dtype, device=image.device).view(-1, 1, 1)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=image.dtype, device=image.device).view(-1, 1, 1)
        image = image / image.max() # [0, 1]
        image = (image - mean) / std # normalize
        return image

    def postprocess(self, d: torch.Tensor):
        dims = (2, 3)
        min_val = d.amin(dim=dims, keepdim=True)
        max_val = d.amax(dim=dims, keepdim=True)
        # Avoid division by zero
        return (d - min_val) / (max_val - min_val + 1e-8)

    def forward(self, image):        
        input = self.preprocess(image)
        result = super().forward(input)[0]
        output = self.postprocess(result)
        return output * 255.0

def main():
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.backends.apple.coreml.partition.coreml_partitioner import CoreMLPartitioner
    # from executorch.backends.qualcomm.qnn.partition.qnn_partitioner import QnnPartitioner
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

    model = Frontend()
    example_inputs = (torch.autograd.Variable(torch.rand((1, 3, 320, 320), dtype=torch.float32)),)

    exported_program = torch.export.export(model, example_inputs)
    program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[
            # CoreMLPartitioner(take_over_mutable_buffer=False)
            VulkanPartitioner(),
            # XnnpackPartitioner()
        ]  # CPU | CoreMLPartitioner() for iOS | QnnPartitioner() for Qualcomm
    ).to_executorch()

    # # 3. Save for deployment
    os.makedirs(os.path.join(root_dirpath, ".results"), exist_ok=True)
    filepath = Path() / ".results" / "fcs_vulkan.pte"
    with open(filepath, "wb") as f:
        f.write(program.buffer)

if __name__ == "__main__":
    main()
    