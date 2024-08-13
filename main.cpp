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
//#include "add.h" // custom op ,if you have one new or different op defination with frame's,please
                   // add head file here.If same with frame , no need to add head file here
#include "subgraph_lhq.h"
#include "scheme_lhq.h"
#include "infershape_lhq.h"
#include <unordered_set>
#include <Python.h>
#include <chrono>
#include <cstdlib>

using namespace std;
using namespace ge;
using ge::Operator;

namespace {
static const int kArgsNum = 6;
static const int kSocVersion = 1;
static const int kGenGraphOpt = 2;
static const int kBeamWidth = 3;
static const int kProtxtPath = 4;
static const int kWeightPath = 5;
static const std::string kPath = "../data/";
}  // namespace

void PrepareOptions(std::map<AscendString, AscendString>& options) {
}

bool CheckIsLHisi(string soc_version) {
    if (soc_version == "Hi3796CV300ES" || soc_version == "Hi3796CV300CS") {
        return true;
    }
    return false;
}

bool GetConstTensorFromBin(string path, Tensor &weight, uint32_t len) {
    ifstream in_file(path.c_str(), std::ios::in | std::ios::binary);
    if (!in_file.is_open()) {
        std::cout << "failed to open" << path.c_str() << '\n';
        return false;
    }
    in_file.seekg(0, ios_base::end);
    istream::pos_type file_size = in_file.tellg();
    in_file.seekg(0, ios_base::beg);

    if (len != file_size) {
        cout << "Invalid Param.len:" << len << " is not equal with binary size��" << file_size << ")\n";
        in_file.close();
        return false;
    }
    char* pdata = new(std::nothrow) char[len];
    if (pdata == nullptr) {
        cout << "Invalid Param.len:" << len << " is not equal with binary size��" << file_size << ")\n";
        in_file.close();
        return false;
    }
    in_file.read(reinterpret_cast<char*>(pdata), len);
    auto status = weight.SetData(reinterpret_cast<uint8_t*>(pdata), len);
    if (status != ge::GRAPH_SUCCESS) {
        cout << "Set Tensor Data Failed"<< "\n";
        delete [] pdata;
        in_file.close();
        return false;
    }
    in_file.close();
    return true;
}
bool GenGraph(Graph& graph) {
    auto shape_data = vector<int64_t>({ 1,1,600,600 });
    TensorDesc desc_data(ge::Shape(shape_data), FORMAT_ND, DT_FLOAT16);

    // data op
    auto data_0 = op::Data("data_0");
    data_0.update_input_desc_x(desc_data);
    data_0.update_output_desc_y(desc_data);
    /*
    // data op
    auto shape_data_1 = vector<int64_t>({ 3,3,1,1 });
    TensorDesc desc_data_1(ge::Shape(shape_data_1), FORMAT_ND, DT_FLOAT16);
    auto data_1 = op::Data("data_1");
    data_1.update_input_desc_x(desc_data_1);
    data_1.update_output_desc_y(desc_data_1);*/
    // custom op ,using method is the same with frame internal op
    // [Notice]: if you want to use custom self-define op, please prepare custom op according to custum op define user guides
    /*auto add = op::Add("add")
        .set_input_x1(data)
        .set_input_x2(data);*/
    // AscendQuant
    /*auto quant = op::AscendQuant("quant")
        .set_input_x(data)
        .set_attr_scale(1.0)
        .set_attr_offset(0.0);
    */
    // const op: conv2d weight
    auto weight_shape = ge::Shape({ 2,2,1,1 });
    TensorDesc desc_weight_1(weight_shape, FORMAT_HWCN, DT_INT8);
    Tensor weight_tensor(desc_weight_1);
    uint32_t weight_1_len = weight_shape.GetShapeSize() * sizeof(int8_t);
    bool res = GetConstTensorFromBin(kPath+"Conv2D_kernel_quant.bin", weight_tensor, weight_1_len);
    if (!res) {
        cout << __LINE__ << "GetConstTensorFromBin Failed!" << endl;
        return -1;
    }
    auto conv_weight = op::Const("Conv2D/weight")
        .set_attr_value(weight_tensor);
   
    // conv2d op
    auto conv2d = op::Conv2D("Conv2d1")
        .set_input_x(data_0)
        .set_input_filter(conv_weight)
        .set_attr_strides({ 2, 2, 2, 2 })
        .set_attr_pads({ 0, 0, 0, 0 })
        .set_attr_dilations({ 1, 1, 1, 1 });

    TensorDesc conv2d_input_desc_x(ge::Shape({1,1,600,600}), FORMAT_NCHW, DT_FLOAT16);
    TensorDesc conv2d_input_desc_filter(ge::Shape({2,2,1,1}), FORMAT_HWCN, DT_FLOAT16);
    TensorDesc conv2d_output_desc_y(ge::Shape({1,1,300,300}), FORMAT_NCHW, DT_FLOAT16);
    conv2d.update_input_desc_x(conv2d_input_desc_x);
    conv2d.update_input_desc_filter(conv2d_input_desc_filter);
    //conv2d.InferShapeAndType();
    conv2d.update_output_desc_y(conv2d_output_desc_y);
    // dequant scale
    /*
    TensorDesc desc_dequant_shape(ge::Shape({ 1 }), FORMAT_ND, DT_UINT64);
    Tensor dequant_tensor(desc_dequant_shape);
    uint64_t dequant_scale_val = 1;
    auto status = dequant_tensor.SetData(reinterpret_cast<uint8_t*>(&dequant_scale_val), sizeof(uint64_t));
    if (status != ge::GRAPH_SUCCESS) {
        cout << __LINE__ << "Set Tensor Data Failed" << "\n";
        return false;
    }
    auto dequant_scale = op::Const("dequant_scale")
        .set_attr_value(dequant_tensor);

    // AscendDequant
    auto dequant = op::AscendDequant("dequant")
        .set_input_x(conv2d)
        .set_input_deq_scale(dequant_scale);
    */
    // const op: BiasAdd weight
    auto weight_bias_add_shape_1 = ge::Shape({ 1 });
    TensorDesc desc_weight_bias_add_1(weight_bias_add_shape_1, FORMAT_ND, DT_FLOAT);
    Tensor weight_bias_add_tensor_1(desc_weight_bias_add_1);
    uint32_t weight_bias_add_len_1 = weight_bias_add_shape_1.GetShapeSize() * sizeof(float);
    float weight_bias_add_value = 0.006448820233345032;
    auto status = weight_bias_add_tensor_1.SetData(reinterpret_cast<uint8_t*>(&weight_bias_add_value), weight_bias_add_len_1);
    if (status != ge::GRAPH_SUCCESS) {
        cout << __LINE__ << "Set Tensor Data Failed" << "\n";
        return false;
    }
    auto bias_weight_1 = op::Const("Bias/weight_1")
        .set_attr_value(weight_bias_add_tensor_1);
    TensorDesc bias_weight_1_output_desc_y(ge::Shape({1}), FORMAT_ND, DT_FLOAT);
    bias_weight_1.update_output_desc_y(bias_weight_1_output_desc_y);
    // BiasAdd 1
    auto bias_add_1 = op::BiasAdd("bias_add_1")
        .set_input_x(conv2d)
        .set_input_bias(bias_weight_1)
        .set_attr_data_format("NCHW");
    TensorDesc bias_add_1_output_desc_y(ge::Shape({1,1,300,300}),FORMAT_NCHW, DT_FLOAT16);
    bias_add_1.update_output_desc_y(bias_add_1_output_desc_y);
    
    // const
    int32_t value[2] = {1,-1};

    auto value_shape = ge::Shape({ 2 });
    TensorDesc desc_dynamic_const(value_shape, FORMAT_ND, DT_INT32);
    Tensor dynamic_const_tensor(desc_dynamic_const);
    uint32_t dynamic_const_len = value_shape.GetShapeSize() * sizeof(int32_t);
    status = dynamic_const_tensor.SetData(reinterpret_cast<uint8_t*>(&(value[0])), dynamic_const_len);
    if (status != ge::GRAPH_SUCCESS) {
        cout << __LINE__ << "Set Tensor Data Failed" << "\n";
        return false;
    }
    auto dynamic_const = op::Const("dynamic_const").set_attr_value(dynamic_const_tensor);
    TensorDesc dynamix_const_output_desc_y(ge::Shape({2}), FORMAT_ND, DT_INT32);
    dynamic_const.update_output_desc_y(dynamix_const_output_desc_y);

    // ReShape op
    auto reshape = op::Reshape("Reshape")
        .set_input_x(bias_add_1)
        .set_input_shape(dynamic_const);
    TensorDesc reshape_output_desc_y(ge::Shape({1,90000}),FORMAT_ND, DT_FLOAT16);
    reshape.update_output_desc_y(reshape_output_desc_y);
    
    // MatMul + BiasAdd
    // MatMul weight 1
    auto matmul_weight_shape_1 = ge::Shape({90000,512});
    TensorDesc desc_matmul_weight_1(matmul_weight_shape_1, FORMAT_ND, DT_FLOAT);
    Tensor matmul_weight_tensor_1(desc_matmul_weight_1);
    uint32_t matmul_weight_1_len = matmul_weight_shape_1.GetShapeSize() * sizeof(float);
    res = GetConstTensorFromBin(kPath + "bias90000.bin", matmul_weight_tensor_1, matmul_weight_1_len);
    if (!res) {
        cout << __LINE__ << "GetConstTensorFromBin Failed!" << endl;
        return -1;
    }
    auto matmul_weight_1 = op::Const("dense/kernel")
        .set_attr_value(matmul_weight_tensor_1);
    matmul_weight_1.update_output_desc_y(desc_matmul_weight_1);
    // MatMul1
    auto matmul_1 = op::MatMul("MatMul_1")
        .set_input_x1(reshape)
        .set_input_x2(matmul_weight_1);
    TensorDesc matmul_1_output_desc_y(ge::Shape({1,512}),FORMAT_ND, DT_FLOAT16);
    matmul_1.update_output_desc_y(matmul_1_output_desc_y);
    
    // BiasAdd const 2
    auto bias_add_shape_2 = ge::Shape({ 512 });
    TensorDesc desc_bias_add_const_1(bias_add_shape_2, FORMAT_ND, DT_FLOAT);
    Tensor bias_add_const_tensor_1(desc_bias_add_const_1);
    uint32_t bias_add_const_len_1 = bias_add_shape_2.GetShapeSize() * sizeof(float);
    res = GetConstTensorFromBin(kPath + "dense_bias.bin", bias_add_const_tensor_1, bias_add_const_len_1);
    if (!res) {
        cout << __LINE__ << "GetConstTensorFromBin Failed!" << endl;
        return -1;
    }
    auto bias_add_const_1 = op::Const("dense/bias")
        .set_attr_value(bias_add_const_tensor_1);
    bias_add_const_1.update_output_desc_y(desc_bias_add_const_1);
    
    // BiasAdd 2
    auto bias_add_2 = op::BiasAdd("bias_add_2")
        .set_input_x(matmul_1)
        .set_input_bias(bias_add_const_1)
        .set_attr_data_format("NCHW");
    TensorDesc bias_add_2_output_desc_y(ge::Shape({1,512}),FORMAT_ND, DT_FLOAT16);
    bias_add_2.update_output_desc_y(bias_add_2_output_desc_y);
    
    // Relu6
    auto relu6 = op::Relu("relu6")
        .set_input_x(bias_add_2);
    relu6.update_output_desc_y(bias_add_2_output_desc_y);
    
    // MatMul weight 2
    auto matmul_weight_shape_2 = ge::Shape({ 512, 10 });
    TensorDesc desc_matmul_weight_2(matmul_weight_shape_2, FORMAT_ND, DT_FLOAT);
    Tensor matmul_weight_tensor_2(desc_matmul_weight_2);
    uint32_t matmul_weight_2_len = matmul_weight_shape_2.GetShapeSize() * sizeof(float);
    res = GetConstTensorFromBin(kPath + "OutputLayer_kernel.bin", matmul_weight_tensor_2, matmul_weight_2_len);
    if (!res) {
        cout << __LINE__ << "GetConstTensorFromBin Failed!" << endl;
        return -1;
    }
    auto matmul_weight_2 = op::Const("OutputLayer/kernel")
        .set_attr_value(matmul_weight_tensor_2);
    matmul_weight_2.update_output_desc_y(desc_matmul_weight_2);
    
    // MatMul 2
    auto matmul_2 = op::MatMul("MatMul_2")
        .set_input_x1(relu6)
        .set_input_x2(matmul_weight_2);
    TensorDesc matmul_2_output_desc_y(ge::Shape({1,10}),FORMAT_ND, DT_FLOAT16);
    matmul_2.update_output_desc_y(matmul_2_output_desc_y);
    
    // BiasAdd const 3
    auto bias_add_shape_3 = ge::Shape({ 10 });
    TensorDesc desc_bias_add_const_3(bias_add_shape_3, FORMAT_ND, DT_FLOAT);
    Tensor bias_add_const_tensor_3(desc_bias_add_const_3);
    uint32_t bias_add_const_len_3 = bias_add_shape_3.GetShapeSize() * sizeof(float);
    res = GetConstTensorFromBin(kPath + "OutputLayer_bias.bin", bias_add_const_tensor_3, bias_add_const_len_3);
    if (!res) {
        cout << __LINE__ << "GetConstTensorFromBin Failed!" << endl;
        return -1;
    }
    auto bias_add_const_3 = op::Const("OutputLayer/bias")
        .set_attr_value(bias_add_const_tensor_3);
    bias_add_const_3.update_output_desc_y(desc_bias_add_const_3);
    // BiasAdd 3
    /*
     * When set input for some node, there are two methodes for you.
     * Method 1: operator level method. Frame will auto connect the node's output edge to netoutput nodes for user
     *   we recommend this method when some node own only one out node
     * Method 2: edge of operator level. Frame will find the edge according to the output edge name
     *   we recommend this method when some node own multi out nodes and only one out edge data wanted back
     */
    auto bias_add_3 = op::BiasAdd("bias_add_3")
        .set_input_x_by_name(matmul_2, "y")
        .set_input_bias_by_name(bias_add_const_3, "y")
        .set_attr_data_format("NCHW");
    TensorDesc bias_add_3_output_desc_y(ge::Shape({1,10}),FORMAT_ND, DT_FLOAT16);
    bias_add_3.update_output_desc_y(bias_add_3_output_desc_y);
    
    // Softmax op
    auto softmax = op::SoftmaxV2("Softmax")
        .set_input_x_by_name(bias_add_3, "y");

    std::vector<Operator> inputs{ data_0 };
    /*
     * The same as set input, when point net output ,Davince framework alos support multi method to set outputs info
     * Method 1: operator level method. Frame will auto connect the node's output edge to netoutput nodes for user
     *   we recommend this method when some node own only one out node
     * Method 2: edge of operator level. Frame will find the edge according to the output edge name
     *   we recommend this method when some node own multi out nodes and only one out edge data wanted back
     * Using method is like follows:
     */
    std::vector<Operator> outputs{ softmax };
    //std::vector<std::pair<ge::Operator, std::string>> outputs_with_name = {{softmax, "y"}};

    graph.SetInputs(inputs).SetOutputs(outputs);

    return true;
}

bool CompareScheme(const SchemeLhq& list_a, const SchemeLhq& list_b) {
	if (list_a.GetPScore() > list_b.GetPScore()) {
		return true;
	} else if (list_a.GetPScore() == list_b.GetPScore()) {
		if (list_a.GetSize() < list_b.GetSize()) return true;
		else return false;
	} else {
		return false;
	}
}

void ShowSubgraphSolution(const vector<SchemeLhq>& subgraph_alter_space) { 
	int solution_num = subgraph_alter_space.size();
	for (int i = 0;i < solution_num;i++) {
		std::cout<<"-----The number:"<<i<< "subgraph partition scheme------"<<std::endl;
		subgraph_alter_space[i].ShowSubgraph();
	}
}

bool SearchGraph(Graph& graph, const int& K = 4) {
    /* TODO Option: If you need to know shape and type info of node, you can write infer shape interface,
     * During the process of traversion nodes, infer operator outputs shapes one by one, and updates outputdesc
     * Attention:(1) new & delete subgraph, deep copy subgraph
     */

    // ignore data, placeholder, const
    std::cout<<"Search Graph Start."<<std::endl;
    
    Py_Initialize();
    if (!Py_IsInitialized()) {
        cout << "python init fail" << endl;
        return 0;
    }
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/home/HwHiAiUser/AscendProject/searchGraph/scripts')");
	  
    vector<SchemeLhq> subgraph_alter_space; // alternate subgraph space
    
    std::vector<GNode> nodes = graph.GetAllNodes();
    graphStatus ret = GRAPH_FAILED;
    for (auto &node : nodes) {
        ge::AscendString name; // get node name
        ret = node.GetName(name);
        if (ret != GRAPH_SUCCESS) {
            std::cout<<"Get node name failed."<<std::endl;
            return false;
        }
        std::string node_name(name.GetString());
        std::cout<<"Find node "<<node_name<<std::endl;

        if (node_name == "data") { // update graph input desc
            TensorDesc desc_data(ge::Shape({1,3,224,224}), FORMAT_NCHW, DT_FLOAT);
            node.UpdateOutputDesc(0, desc_data);
        }

        //InferShapeLhq ifl;
        //ifl.InferOutputDesc(node);

        ge::AscendString type;  //  get node type
        ret = node.GetType(type);
        if (ret != GRAPH_SUCCESS) {
            std::cout<<"Get node type failed."<<std::endl;
            return false;
        }
        std::string node_type(type.GetString());
        
        const std::unordered_set<std::string> ignore_set = {"Const", "Data", "PlaceHolder", "Pooling"}; // don't handle these type
        if (ignore_set.find(node_type) != ignore_set.end()) {
            continue;
        }
        auto feature_optype = cann2feature_optype.find(node_type);
        if (feature_optype == cann2feature_optype.end()) { // not find, don't handle
            cout<<"CANN OP type: "<<node_type<<" is not defined in cann2feature_optype"<<endl;
            continue;
        }
	   
        if (subgraph_alter_space.empty()) { // the first node
        	SchemeLhq scheme0;
        	SubgraphLhqPtr sgraph0 = std::make_shared<SubgraphLhq>(node);
        	scheme0.AddSubgraph(sgraph0);
        	subgraph_alter_space.push_back(scheme0);
        	continue;
	    } 

 		vector<SchemeLhq> tem_alter_space; // the alternate spaced of this cycle
		for (int i = 0;i < subgraph_alter_space.size();i++) {
            subgraph_alter_space[i].AddNode(node, tem_alter_space); // tem_alter_space accumulate, not clear
			//cout<<"alter_space size"<<tem_alter_space.size()<<endl;
		} 
		
		// sort by pScore
		std::sort(tem_alter_space.begin(), tem_alter_space.end(), CompareScheme);
		if (tem_alter_space.size() < K) {
			subgraph_alter_space.assign(tem_alter_space.begin(), tem_alter_space.end());
		} else {
			subgraph_alter_space.assign(tem_alter_space.begin(), tem_alter_space.begin() + K);
		}
		
    }
   
    std::cout<<"Search Graph Success."<<std::endl;
    ShowSubgraphSolution(subgraph_alter_space);
    
    Py_Finalize();

    return true;
}

int main(int argc, char* argv[])
{
    cout << "========== Test Start ==========" << endl;
    if (argc != kArgsNum) {
        cout << "[ERROR]input arg num must be 5! " << endl;
        cout << "The second arg stand for soc version! Please retry with your soc version " << endl;
        cout << "[Notice] Supported soc version as list:Ascend310 Ascend910 Ascend610 Ascend620 Hi3796CV300ES Hi3796CV300CS" << endl;
        cout << "The third arg stand for Generate Graph Options! Please retry with your soc version " << endl;
        cout << "[Notice] Supported Generate Graph Options as list:" << endl;
        cout << "    [gen]: GenGraph" << endl;
        cout << "    [tf]: generate from tensorflow origin model;" << endl;
        cout << "    [caffe]: generate from caffe origin model" << endl;
        return -1;
    }
    cout << argv[kSocVersion] << endl;
    cout << argv[kGenGraphOpt] << endl;
    cout << argv[kBeamWidth] << endl;
    cout << argv[kProtxtPath] << endl;
    cout << argv[kWeightPath] << endl;

    // 1. Genetate graph
    Graph graph1("IrGraph1");
    bool ret;

    if (string(argv[kGenGraphOpt]) == "gen") {
        ret = GenGraph(graph1);
        if (!ret) {
            cout << "========== Generate Graph1 Failed! ==========" << endl;
            return -1;
        }
        else {
            cout << "========== Generate Graph1 Success! ==========" << endl;
        }
    } else if (string(argv[kGenGraphOpt]) == "tf") {
        std::string tfPath = "/home/HwHiAiUser/modelzoo/yolov5s/yolov5s.pb";
        graphStatus tfStatus = GRAPH_SUCCESS;

        if (CheckIsLHisi(string(argv[kSocVersion]))) {
            std::map<AscendString, AscendString> parser_options = {
                {AscendString(ge::ir_option::INPUT_FP16_NODES), AscendString("input1;input2")}
            };
            tfStatus = ge::aclgrphParseTensorFlow(tfPath.c_str(), parser_options, graph1);
        } else {
            tfStatus = ge::aclgrphParseTensorFlow(tfPath.c_str(), graph1);
        }
        if (tfStatus != GRAPH_SUCCESS) {
            cout << "========== Generate graph from tensorflow origin model failed.========== " << endl;
            return 0;
        }
        cout << "========== Generate graph from tensorflow origin model success.========== " << endl;
        
    } else if (string(argv[kGenGraphOpt]) == "caffe") {
        std::string caffePath = string(argv[kProtxtPath]);
        std::string weigtht = string(argv[kWeightPath]);
        graphStatus caffeStatus = GRAPH_SUCCESS;
        if (CheckIsLHisi(string(argv[kSocVersion]))) {
            std::map<AscendString, AscendString> parser_options = {
                {AscendString(ge::ir_option::INPUT_FP16_NODES), AscendString("data")}
            };
            caffeStatus = ge::aclgrphParseCaffe(caffePath.c_str(), weigtht.c_str(), parser_options, graph1);
        } else {
            caffeStatus = ge::aclgrphParseCaffe(caffePath.c_str(), weigtht.c_str(), graph1);
        }
        if (caffeStatus != GRAPH_SUCCESS) {
            cout << "========== Generate graph from caffe origin model failed.========== " << endl;
            return 0;
        }
        cout << "========== Generate graph from caffe origin model success.========== " << endl;
    } 
    
    auto start = std::chrono::high_resolution_clock::now();
    SearchGraph(graph1, atoi(argv[kBeamWidth]));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
    
    /*
    // 2. system init
    std::map<AscendString, AscendString> global_options = {
        {AscendString(ge::ir_option::SOC_VERSION), AscendString(argv[kSocVersion])}  ,
    };
    auto status = aclgrphBuildInitialize(global_options);
    // 3. Build Ir Model1
    ModelBufferData model1;
    std::map<AscendString, AscendString> options;
    PrepareOptions(options);

    status = aclgrphBuildModel(graph1, options, model1);
    if (status == GRAPH_SUCCESS) {
        cout << "Build Model1 SUCCESS!" << endl;
    }
    else {
        cout << "Build Model1 Failed!" << endl;
    }
    // 4. Save Ir Model
    status = aclgrphSaveModel("model_test", model1);
    if (status == GRAPH_SUCCESS) {
        cout << "Save Offline Model1 SUCCESS!" << endl;
    }
    else {
        cout << "Save Offline Model1 Failed!" << endl;
    }

    // release resource
    aclgrphBuildFinalize();
*/
    return 0;
}
