#include <iostream>
#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "attr_value.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_ir_build.h"
#include "all_ops.h"
#include <vector>
#include "infershape_lhq.h"

using namespace std;
using namespace ge;
using ge::Operator;

void InferShapeLhq::InferOutputDesc(GNode& node) {
    ge::AscendString type;  //  get op type
    graphStatus ret = node.GetType(type);
    if (ret != GRAPH_SUCCESS) {
        std::cout<<"Get node type failed."<<std::endl;
        return;
    }
    std::string node_type(type.GetString());
    if (node_type == "Conv2D") {
        this->InferOutputDescConv2D(node);
    }
}

void InferShapeLhq::InferOutputDescConv2D(GNode& node) {
    graphStatus ret;
    TensorDesc x_desc; // get input desc
    ret = node.GetInputDesc(0, x_desc);
    if (ret != GRAPH_SUCCESS) {
        std::cout<<"Get node input x desc failed."<<std::endl;
        return;
    }

    int64_t N, height, width, in_channel, out_channel, kernel_height, kernel_width;
    // input demensions
    Format x_format = x_desc.GetFormat();
    if (x_format == FORMAT_NCHW) {
        N = x_desc.GetShape().GetDim(0);
        height = x_desc.GetShape().GetDim(2);
        width = x_desc.GetShape().GetDim(3);
        in_channel = x_desc.GetShape().GetDim(1);
    } else if (x_format == FORMAT_NHWC) {
        N = x_desc.GetShape().GetDim(0);
        height = x_desc.GetShape().GetDim(1);
        width = x_desc.GetShape().GetDim(2);
        in_channel = x_desc.GetShape().GetDim(3);
    }

    Tensor filter_tensor; // get filter desc
    ret = node.GetInputConstData(1, filter_tensor);
    TensorDesc filter_desc = filter_tensor.GetTensorDesc();

    // kernel demensions
    Format filter_format = filter_desc.GetFormat();
    if (filter_format == FORMAT_NCHW) {
        kernel_height = filter_desc.GetShape().GetDim(2);
        kernel_width = filter_desc.GetShape().GetDim(3);
        out_channel = filter_desc.GetShape().GetDim(0);
        if (in_channel != filter_desc.GetShape().GetDim(1)) {
            std::cout<<"Infer shape failled: filter's in_channel is different from x's in_channel"<<std::endl;
            return;
        }
    } else if (filter_format == FORMAT_HWCN) {
        kernel_height = filter_desc.GetShape().GetDim(0);
        kernel_width = filter_desc.GetShape().GetDim(1);
        out_channel = filter_desc.GetShape().GetDim(3);
        if (in_channel != filter_desc.GetShape().GetDim(2)) {
            std::cout<<"Infer shape failled: filter's in_channel is different from x's in_channel"<<std::endl;
            return;
        }
    }

    std::vector<int64_t> pads, strides; // get pads & strides
    node.GetAttr(AscendString("strides"), strides);
    node.GetAttr(AscendString("pads"), strides);
    // Calculate output dimensions
    int64_t out_height = (height - kernel_height + pads[0] + pads[1]) / strides[2] + 1;
    int64_t out_width = (width - kernel_width + pads[2] + pads[3]) / strides[3] + 1;

    std::vector<int64_t> outputShape;
    if (x_format == FORMAT_NCHW) {
        outputShape = {N, out_channel, out_height, out_width};
    } else { // NHWC
        outputShape = {N, out_height, out_width, out_channel};
    }


}

