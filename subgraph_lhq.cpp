#include <iostream>
#include <fstream>
#include <algorithm>
#include <string.h>
#include "tensorflow_parser.h"
#include "caffe_parser.h"
#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "attr_value.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_ir_build.h"
#include "all_ops.h"
#include <dlfcn.h>
#include <unistd.h>
#include <cstdint>

#include <unordered_map>
#include <vector>
#include <memory>
#include <stdexcept>
#include <Python.h>
#include "subgraph_lhq.h"

using namespace std;
using namespace ge;
using ge::Operator;


void vector2PyList(const vector < int >& src, PyObject* dst) {
    if (dst == nullptr) {
        return ;
    }

    for (int x: src) {
        PyObject* tem = PyLong_FromLong(x);
        if (tem != nullptr) {
            PyList_Append(dst, tem);
        } else {
            PyErr_Print();
            return ;
        }
    }
}

bool isEqualNode(const GNodePtr& a, const GNodePtr& b) {
    ge::AscendString name;
    graphStatus ret = a->GetName(name);
    if (ret != GRAPH_SUCCESS) {
        std::cout << "Get node name failed." << std::endl;
        return false;
    }
    std::string a_name(name.GetString());

    ret = b->GetName(name);
    if (ret != GRAPH_SUCCESS) {
        std::cout << "Get node name failed." << std::endl;
        return false;
    }
    std::string b_name(name.GetString());

    return a_name == b_name ? true : false;
}

bool isEqualTensorDesc(const TensorDesc& a, const TensorDesc& b) {
    if (a.GetFormat() != b.GetFormat()) return false;
    if (a.GetDataType() != b.GetDataType()) return false;
    Shape shape_a = a.GetShape();
    Shape shape_b = b.GetShape();
    if (shape_a.GetDims() != shape_b.GetDims()) return false;
    return true;
}

SubgraphLhq::SubgraphLhq(GNode input0) {
    graphStatus ret;
    std::string node_type = this->GetNodeType(input0);
    // abandon unrecorded types
    auto feature_optype = cann2feature_optype.find(node_type);
    if (feature_optype == cann2feature_optype.end()) {
        // not find
        cout << "CANN OP type: " << node_type << "is not defined in cann2feature_optype" << endl;
        throw std::invalid_argument("Node type must in feature list.");
    }

    // set node_list
    GNodePtr pnode = std::make_shared <GNode>(input0);
    this->node_list.push_back(pnode);

    // set input
    size_t index_num = input0.GetInputsSize();
    for (size_t i = 0; i < index_num; i++) {
        Tensor input_tensor;
        ret = input0.GetInputConstData(static_cast<int32_t>(i), input_tensor);
        if (ret != GRAPH_SUCCESS) { // not constant
            this->input_nodes.push_back({pnode, static_cast<int32_t>(i)});
        } else { // is constant
            TensorDesc tensor_desc = input_tensor.GetTensorDesc();
            this->input_constant.push_back(tensor_desc);
        }
    }

    // set output desc
    index_num = input0.GetOutputsSize();
    for (size_t i = 0; i < index_num; i++) {
        int connection_num = input0.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(i)).size();
        for (int j = 0;j < connection_num;j++) this->output_nodes.push_back({pnode, static_cast<int32_t>(i)});
    }

}

SubgraphLhq::SubgraphLhq(const SubgraphLhq& other) {
    this->node_list = other.node_list;
    this->input_nodes = other.input_nodes;
    this->output_nodes = other.output_nodes;
    this->input_constant = other.input_constant;
    this->pScore = other.pScore;
    this->dpCoff = other.dpCoff;
    // use deep copy, the son and fathter need to be redecide, not here
    //this->father_subgraph = other.father_subgraph;
    //this->son_subgraph = other.son_subgraph;
}

double SubgraphLhq::GetPScore() const {
    return this->pScore;
}

int SubgraphLhq::GetSize() const {
    return this->node_list.size();
}

bool SubgraphLhq::operator<(const SubgraphLhq& other) const {
    if (this->pScore < other.GetPScore()) return true;
    else return false;
}

bool SubgraphLhq::operator==(const SubgraphLhq& other) const {
    if (this->node_list.size() != other.node_list.size()) return false;
    for (int i = 0;i < this->node_list.size();i++) {
        if (!isEqualNode(this->node_list[i], other.node_list[i])) return false;
    }
    return true;
}

bool SubgraphLhq::ContainNode(const GNodePtr& node) const {
    ge::AscendString name;
    graphStatus ret = node->GetName(name);
    if (ret != GRAPH_SUCCESS) {
        std::cout << "Get node name failed." << std::endl;
        return false;
    }
    std::string node_name(name.GetString());
    //std::cout<<"Is contain node "<<node_name<<"?"<<std::endl;
    for (auto x : this->node_list) {
        ret = x->GetName(name);
        if (ret != GRAPH_SUCCESS) {
            std::cout << "Get node name failed." << std::endl;
            return false;
        }
        std::string x_node_name(name.GetString());
        //std::cout<<"Node "<<x_node_name<<"in subgraph"<<std::endl;
        if (x_node_name == node_name) return true;
    }
    return false;
}

std::vector<std::pair<GNodePtr, int32_t>>::iterator
    SubgraphLhq::FindInput(const std::pair<GNodePtr, int32_t>& input) {

    for (auto it = this->input_nodes.begin(); it != this->input_nodes.end();it++) {
        if (isEqualNode(it->first, input.first) && it->second == input.second) return it;
    }
    return this->input_nodes.end();
}

std::vector<std::pair<GNodePtr, int32_t>>::iterator
    SubgraphLhq::FindOutput(const std::pair<GNodePtr, int32_t>& output) {

    for (auto it = this->output_nodes.begin(); it != this->output_nodes.end();it++) {
        if (isEqualNode(it->first, output.first) && it->second == output.second)
            return it;
    }
    return this->output_nodes.end();
}

bool SubgraphLhq::LoopCheck() {
    for (auto it_i = this->input_nodes.begin(); it_i != this->input_nodes.end();it_i++) {
        std::pair<GNodePtr, int32_t> vi_node = it_i->first->GetInDataNodesAndPortIndexs(it_i->second);
        for (auto it_o = this->output_nodes.begin();it_o != this->output_nodes.end();it_o++) {
            vector<std::pair<GNodePtr, int32_t>> vo_nodes = it_o->first->GetOutDataNodesAndPortIndexs(it_o->second);
            for (auto vo_node: vo_nodes) {
                if (isEqualNode(vo_node.first, vi_node.first)) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool SubgraphLhq::ContainOutput(const std::pair<GNodePtr, int32_t>& output) const {
    for (auto it = this->output_nodes.begin(); it != this->output_nodes.end();it++) {
        if (isEqualNode(it->first, output.first) && it->second == output.second)
        return true;
    }
    return false;
}

string SubgraphLhq::GetNodeType(const GNode& node) const {
    ge::AscendString type;
    graphStatus ret = node.GetType(type);
    if (ret != GRAPH_SUCCESS) {
        std::cout << "Get node type failed." << std::endl;
        return "";
    }
    std::string node_type(type.GetString());

    // conv2d -> 1x1, 3x3, 5x5
    if (node_type == "Conv2D") {
        Tensor filter_data; // Get shape of filter
        ret = node.GetInputConstData(1, filter_data);
        TensorDesc filter_desc = filter_data.GetTensorDesc();
        if (filter_desc.GetFormat() == FORMAT_NCHW) {
            int filter_hw = filter_desc.GetShape().GetDim(2);
            node_type = node_type + std::to_string(filter_hw) + "x" + std::to_string(filter_hw);
        } else if (filter_desc.GetFormat() == FORMAT_HWCN) {
            int filter_hw = filter_desc.GetShape().GetDim(0);
            node_type = node_type + std::to_string(filter_hw) + "x" + std::to_string(filter_hw);
        } else if (filter_desc.GetFormat() == FORMAT_ND) {
            int filter_0 = filter_desc.GetShape().GetDim(0);
            int filter_1 = filter_desc.GetShape().GetDim(1);
            int filter_2 = filter_desc.GetShape().GetDim(2);
            int filter_3 = filter_desc.GetShape().GetDim(3);
            if (filter_0 == filter_1 && filter_0 < 8) {
                int filter_hw = filter_0;
                node_type = node_type + std::to_string(filter_hw) + "x" + std::to_string(filter_hw);
            } else if (filter_2 == filter_3 && filter_2 < 8) {
                int filter_hw = filter_2;
                node_type = node_type + std::to_string(filter_hw) + "x" + std::to_string(filter_hw);
            } else {
                std::cout<< "Conv filter format is ND, but can't get valid shape, shape is "
                    <<filter_0<<" "<<filter_1<<" "<<filter_2<<" "<<filter_3<<"."<<std::endl;
                return "";
            }
        } else {
            std::cout<< "Conv filter format is wrong, neither NCHW nor HWCN."<<std::endl;
            return "";
        }
    }
    return node_type;
}

bool SubgraphLhq::AddNode(const GNode & node) {
    graphStatus ret;
    std::string node_type = this->GetNodeType(node);

    // abandon unrecorded types
    auto feature_optype = cann2feature_optype.find(node_type);
    if (feature_optype == cann2feature_optype.end()) {
        // not find
        cout << "CANN OP type: " << node_type << "is not defined in cann2feature_optype" << endl;
        return false;
    }

    // add node list
    GNodePtr pnode = std::make_shared<GNode>(node);
    this->node_list.push_back(pnode);

    // set input
    size_t input_num = node.GetInputsSize();
    for (size_t i = 0;i < input_num;i++) {
        // desc is obtained differently depending on whether it is a constant
        Tensor input_tensor;
        ret = node.GetInputConstData(static_cast<int32_t>(i), input_tensor);
        if (ret != GRAPH_SUCCESS) { // not constant
            std::pair<GNodePtr, int32_t> pre_node = node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
            auto pos = this->FindOutput(pre_node);
            if (pos == this->output_nodes.end()) { // add new input
                this->input_nodes.push_back({pnode, static_cast<int32_t>(i)});
            } else { // remove origin output
                this->output_nodes.erase(pos);
            }
        } else { // is constant
            TensorDesc tensor_desc = input_tensor.GetTensorDesc();
            // do not consider small shape constant, like 'mean' in bnin or 'shape' in reshape
            int64_t const_shape_size = tensor_desc.GetShape().GetShapeSize();
            if (const_shape_size > 5) this->input_constant.push_back(tensor_desc);
        }
    }
    // don't need to remove origin input, because according to the topological order, the newly added node
    // cannot be the input of any node in the subgraph

    // set output
    // must be output node ,cause topological orider，none of the outputs of it is added
    size_t output_num = node.GetOutputsSize();
    for (size_t i = 0;i < output_num;i++) {
        // topological order, output must not in graph
        int connection_num = node.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(i)).size();
        for (int j = 0;j < connection_num;j++) this->output_nodes.push_back({pnode, static_cast<int32_t>(i)});
    }
    if (this->LoopCheck() == false) return false;
    // update PScore
    this->UpdatePScore();
    return true;
}

void SubgraphLhq::Merge(const SubgraphLhqPtr& other) {
    // set node list
    // TODO keep topological order, father subgraph first, brothers doesn't matter
    this->node_list.insert(this->node_list.end(), other->node_list.begin(), other->node_list.end());
   
    // set input & output
    // find where input overlaps with output, because no ring in graph, either father-son or brother relation
    vector<std::pair<GNodePtr, int32_t>> copy_other_output(other->output_nodes);
    for (int i = this->input_nodes.size() - 1;i >= 0;i--) {
        GNodePtr input_x = this->input_nodes[i].first;
        std::pair<GNodePtr, int32_t> input_this = input_x->GetInDataNodesAndPortIndexs(this->input_nodes[i].second);
        for (int j = copy_other_output.size() - 1;j >= 0;j--) {
            auto output_other = copy_other_output[j];
            if (isEqualNode(input_this.first, output_other.first) && input_this.second == output_other.second) {
                this->input_nodes.erase(this->input_nodes.begin() + i);
                copy_other_output.erase(copy_other_output.begin() + j);
                break; // one input one output
            }
        }
    }
   
    vector<std::pair<GNodePtr, int32_t>> copy_other_input(other->input_nodes);
    for (int i = copy_other_input.size() - 1;i >= 0;i--) {
        GNodePtr input_x = copy_other_input[i].first;
        std::pair<GNodePtr, int32_t> input_other = input_x->GetInDataNodesAndPortIndexs(copy_other_input[i].second);
        for (int j = this->output_nodes.size() - 1;j >= 0;j--) {
            auto output_this = this->output_nodes[j];
            if (isEqualNode(input_other.first, output_this.first) && input_other.second == output_this.second) {
                copy_other_input.erase(copy_other_input.begin() + i);
                this->output_nodes.erase(this->output_nodes.begin() + j);
                break; // one input one output
            }
        }
    }
    this->input_nodes.insert(this->input_nodes.end(), copy_other_input.begin(), copy_other_input.end());
    this->output_nodes.insert(this->output_nodes.end(), copy_other_output.begin(), copy_other_output.end());
   
    // set constant
    this->input_constant.insert(this->input_constant.end(), other->input_constant.begin(), other->input_constant.end());
    
    // update Pscore
    this->UpdatePScore();
    
}

bool SubgraphLhq::Feature2Libsvm(vector<int>& col, vector<int>& data) {
    col.clear();
    data.clear();
    
    graphStatus ret;
    // optypes & category
    for (int i = 0;i < this->node_list.size();i++) {
        col.push_back(i + 1);
        std::string node_type = this->GetNodeType(*(this->node_list[i]));
        auto feature_optype = cann2feature_optype.find(node_type); // op_type
        if (feature_optype == cann2feature_optype.end()) {
            // not find
            cout << "CANN OP type: " << node_type << " is not defined in cann2feature_optype. Feature2Libsvm failed." << endl;
            return false;
        }
        data.push_back(static_cast<int>(feature_optype->second));
   
        auto feature_opcate = cann2feature_category.find(node_type); // op category
        if (feature_opcate == cann2feature_category.end()) {
            cout << "CANN OP type: " << node_type << " is not defined in cann2feature_category. Feature2Libsvm failed." << endl;
            return false;
        }
        auto pos = std::find(col.begin(), col.end(), static_cast<int>(feature_opcate->second));
        if (pos == col.end()) { // first appearance       
            col.push_back(static_cast<int>(feature_opcate->second));
            data.push_back(1);
        } else { // second or more
            data[pos-col.begin()]++;
        }
        
    }
    // input desc
    if (this->input_nodes.size() > 5) {
        std::cout << "Subgraph input size is bigger than 5, cost model is not available." << std::endl;
        return false;
    }
    for (int i = 0;i < this->input_nodes.size();i++) { // input nodes
        std::pair<GNodePtr, int32_t> input_i = this->input_nodes[i];
        TensorDesc desc_i;
        ret = input_i.first->GetInputDesc(input_i.second, desc_i);
        // format
        col.push_back(51 + i * 7);
        Format f_i = desc_i.GetFormat();
        auto tem_f = cann2feature_format.find(f_i);
        if (tem_f == cann2feature_format.end()) {
            cout << "Format is not entered and will be set as default." << endl;
            data.push_back(0);
        } else {
            data.push_back(static_cast<int>(tem_f->second));
        }
        // shape
        Shape s_i = desc_i.GetShape();
        size_t dim_num = s_i.GetDimNum();
        //cout << "dim num is" << dim_num << endl;
        //cout << "shape size is" << s_i.GetShapeSize()<< endl;
        if (dim_num > 5) {
            cout << "Input dim number is bigger than 6, cost model is no available." << endl;
            return false;
        }
        for (int j = 0;j < 5;j++) {
            col.push_back(52 + i * 7 + j);
            if (j < dim_num) data.push_back(s_i.GetDim(j));
            else data.push_back(1);
        }
        // datatype
        col.push_back(57 + i * 7);
        DataType d_i = desc_i.GetDataType();
        auto tem_d = cann2feature_datatype.find(d_i);
        if (tem_d == cann2feature_datatype.end()) {
            cout << "DataType is not entered and will be set as default." << endl;
            data.push_back(0);
        } else {
            data.push_back(static_cast<int>(tem_d->second));
        }
    }
    int feature_cnum = 5 - this->input_nodes.size(); // the number of constant use to predict
    int record_cnum = 0;
    for (int i = 0;i < this->input_constant.size() && record_cnum < feature_cnum;i++) { // input constant
        TensorDesc desc_i = this->input_constant[i];
        if (i > 0 && isEqualTensorDesc(desc_i, this->input_constant[i-1])) { // Do not enter duplicate constants
            continue;
        }
        // format
        col.push_back(51 + (this->input_nodes.size() + record_cnum) * 7);
        Format f_i = desc_i.GetFormat();
        auto tem_f = cann2feature_format.find(f_i);
        if (tem_f == cann2feature_format.end()) {
            cout << "Format is not entered and will be set as default." << endl;
            data.push_back(0);
        } else {
            data.push_back(static_cast<int>(tem_f->second));
        }
        // shape
        Shape s_i = desc_i.GetShape();
        size_t dim_num = s_i.GetDimNum();
        //cout << "dim num is" << dim_num << endl;
        //cout << "shape size is" << s_i.GetShapeSize()<< endl;
        if (dim_num > 5) {
            cout << "Input dim number is bigger than 6, cost model is no available." << endl;
            return false;
        }
        for (int j = 0;j < 5;j++) {
            col.push_back(52 + (this->input_nodes.size() + record_cnum) * 7 + j);
            if (j < dim_num) data.push_back(s_i.GetDim(j));
            else data.push_back(1);
        }
        // datatype
        col.push_back(57 + (this->input_nodes.size() + record_cnum) * 7);
        DataType d_i = desc_i.GetDataType();
        auto tem_d = cann2feature_datatype.find(d_i);
        if (tem_d == cann2feature_datatype.end()) {
            cout << "DataType is not entered and will be set as default." << endl;
            data.push_back(0);
        } else {
            data.push_back(static_cast<int>(tem_d->second));
        }
        record_cnum++;
    }

    return true;
}

double SubgraphLhq::UpdatePScore() {
/*
  Handle special cases:
  1) Reshape cannot merge, due to interface limitations
  2) One node is 0
*/
    if (this->node_list.size() == 1) {
        this->pScore = 0.0;
        return 0;
    }
// Start to update score
    PyObject * pModule = PyImport_ImportModule("predict");
    if (pModule == NULL) {
        PyErr_Print();
        cout << "module not found" << endl;
        return 1;
    }
    PyObject * pFunc = PyObject_GetAttrString(pModule, "predict_time");
    if (!pFunc || !PyCallable_Check(pFunc)) {
        cout << "not found function predict_time" << endl;
        return 0;
    }
    // transfer feature to svm format
    vector<int> col, data;
    bool res = Feature2Libsvm(col, data);
    if (!res) {
        cout << "transfer feature to libsvm failed." << endl;
    }
    
    // print col, data
    cout << "feature in libsvm:" << endl;
    for (size_t j = 0;j < col.size();j++) {
        cout << col[j] << ":" << data[j] << " ";
    }
    cout << endl;
    
    PyObject * list1 = PyList_New(0);
    PyObject * list2 = PyList_New(0);
    vector2PyList(col, list1);
    vector2PyList(data, list2);
    PyObject * pArgs = PyTuple_Pack(2, list1, list2);

    PyObject * pResult = PyObject_CallObject(pFunc, pArgs);
    double score = PyFloat_AsDouble(pResult);
    /*
    // change mode to dsl
    PyList_SetSlice(list2, col.size()- 1, col.size(), NULL);
    PyList_Append(list2, PyLong_FromLong(3));
    pArgs = PyTuple_Pack(2, list1, list2);
    pResult = PyObject_CallObject(pFunc, pArgs);
    double dsl_score = PyFloat_AsDouble(pResult);
    */

    this->pScore = score;
    cout<< "pScore:" << this->pScore << endl;

    Py_DECREF(pArgs);
    Py_DECREF(list1);
    Py_DECREF(list2);
    Py_XDECREF(pResult);
    Py_DECREF(pFunc);
    Py_DECREF(pModule);

    return this->pScore;
}

void SubgraphLhq::ShowNodes() const {
    for (const GNodePtr & pnode : node_list) {
        ge::AscendString name;
        graphStatus ret = pnode->GetName(name);
        if (ret != GRAPH_SUCCESS) {
            std::cout << "Get node name failed in SubgraphLhq::ShowNodes." << std::endl;
        }
        std::string node_name(name.GetString());
        std::cout << node_name << " " << std::endl;
    }
    std::cout << std::endl;
}
