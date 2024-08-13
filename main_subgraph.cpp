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
#include <unordered_set>
#include <Python.h>

using namespace std;
using namespace ge;
using ge::Operator;

namespace {
static const int kArgsNum = 3;
static const int kSocVersion = 1;
static const int kGenGraphOpt = 2;
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

Graph BuildSubgraph() {
    Graph subgraph("MmBiasSoftmax");
    
    auto shape_sub_data_0 = vector<int64_t>({1, 512});
    TensorDesc desc_sub_data_0(ge::Shape(shape_sub_data_0), FORMAT_ND, DT_FLOAT16);
    auto data_0_subgraph = op::Data("Data_0_Subgraph").set_attr_index(0);
    data_0_subgraph.update_input_desc_x(desc_sub_data_0);
    data_0_subgraph.update_output_desc_y(desc_sub_data_0);
    
    // MatMul weight 2
    auto matmul_weight_shape_2 = ge::Shape({ 512, 10 });
    TensorDesc desc_matmul_weight_2(matmul_weight_shape_2, FORMAT_ND, DT_FLOAT);
    Tensor matmul_weight_tensor_2(desc_matmul_weight_2);
    uint32_t matmul_weight_2_len = matmul_weight_shape_2.GetShapeSize() * sizeof(float);
    auto res = GetConstTensorFromBin(kPath + "OutputLayer_kernel.bin", matmul_weight_tensor_2, matmul_weight_2_len);
    if (!res) {
        cout << __LINE__ << "GetConstTensorFromBin Failed!" << endl;
        return subgraph;
    }
    auto matmul_weight_2 = op::Const("OutputLayer/kernel")
        .set_attr_value(matmul_weight_tensor_2);
    matmul_weight_2.update_output_desc_y(desc_matmul_weight_2);
    
    // MatMul 2
    auto matmul_2 = op::MatMul("MatMul_2")
        .set_input_x1(data_0_subgraph)
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
        return subgraph;
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
        
    std::vector<Operator> inputs_subgraph{ data_0_subgraph };
    std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(softmax, vector<std::size_t>{0});
    subgraph.SetInputs(inputs_subgraph).SetOutputs(output_indexs);
    return subgraph;

    
}

bool GenGraph(Graph& graph)
{
    auto shape_data = vector<int64_t>({ 1,1,28,28 });
    TensorDesc desc_data(ge::Shape(shape_data), FORMAT_ND, DT_FLOAT16);

    // data op
    auto data_0 = op::Data("data_0");
    data_0.update_input_desc_x(desc_data);
    data_0.update_output_desc_y(desc_data);
    
    // data op
    auto shape_data_1 = vector<int64_t>({ 2,2,1,1 });
    TensorDesc desc_data_1(ge::Shape(shape_data_1), FORMAT_ND, DT_FLOAT16);
    auto data_1 = op::Data("data_1");
    data_1.update_input_desc_x(desc_data_1);
    data_1.update_output_desc_y(desc_data_1);
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

    // const op: conv2d weight
    auto weight_shape = ge::Shape({ 2,2,1,1 });
    TensorDesc desc_weight_1(weight_shape, FORMAT_ND, DT_INT8);
    Tensor weight_tensor(desc_weight_1);
    uint32_t weight_1_len = weight_shape.GetShapeSize() * sizeof(int8_t);
    bool res = GetConstTensorFromBin(kPath+"Conv2D_kernel_quant.bin", weight_tensor, weight_1_len);
    if (!res) {
        cout << __LINE__ << "GetConstTensorFromBin Failed!" << endl;
        return -1;
    }
    auto conv_weight = op::Const("Conv2D/weight")
        .set_attr_value(weight_tensor);
    */
    // conv2d op
    auto conv2d = op::Conv2D("Conv2d1")
        .set_input_x(data_0)
        .set_input_filter(data_1)
        .set_attr_strides({ 1, 1, 1, 1 })
        .set_attr_pads({ 0, 1, 0, 1 })
        .set_attr_dilations({ 1, 1, 1, 1 });

    TensorDesc conv2d_input_desc_x(ge::Shape({1,1,28,28}), FORMAT_NCHW, DT_FLOAT16);
    TensorDesc conv2d_input_desc_filter(ge::Shape({2,2,1,1}), FORMAT_HWCN, DT_FLOAT16);
    TensorDesc conv2d_output_desc_y(ge::Shape({1,1,28,28}), FORMAT_NCHW, DT_FLOAT16);
    conv2d.update_input_desc_x(conv2d_input_desc_x);
    conv2d.update_input_desc_filter(conv2d_input_desc_filter);
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
    TensorDesc bias_add_1_output_desc_y(ge::Shape({1,1,28,28}),FORMAT_NCHW, DT_FLOAT16);
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
    TensorDesc reshape_output_desc_y(ge::Shape({1,784}),FORMAT_ND, DT_FLOAT16);
    reshape.update_output_desc_y(reshape_output_desc_y);
    
    // MatMul + BiasAdd
    // MatMul weight 1
    auto matmul_weight_shape_1 = ge::Shape({784,512});
    TensorDesc desc_matmul_weight_1(matmul_weight_shape_1, FORMAT_ND, DT_FLOAT);
    Tensor matmul_weight_tensor_1(desc_matmul_weight_1);
    uint32_t matmul_weight_1_len = matmul_weight_shape_1.GetShapeSize() * sizeof(float);
    bool res = GetConstTensorFromBin(kPath + "dense_kernel.bin", matmul_weight_tensor_1, matmul_weight_1_len);
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
    
    SubgraphBuilder subgraphBuilder_0 = BuildSubgraph;
    auto mm_bias_softmax = op::PartitionedCall("mmBiasSoftmax")
        .create_dynamic_input_byindex_args(1, 0)
        .set_dynamic_input_args(0, relu6)
        .set_subgraph_builder_f(subgraphBuilder_0)
        .create_dynamic_output_output(1);
    std::vector<Operator> inputs{ data_0, data_1 };
    /*
     * The same as set input, when point net output ,Davince framework alos support multi method to set outputs info
     * Method 1: operator level method. Frame will auto connect the node's output edge to netoutput nodes for user
     *   we recommend this method when some node own only one out node
     * Method 2: edge of operator level. Frame will find the edge according to the output edge name
     *   we recommend this method when some node own multi out nodes and only one out edge data wanted back
     * Using method is like follows:
     */
    std::vector<Operator> outputs{ mm_bias_softmax };
    //std::vector<std::pair<ge::Operator, std::string>> outputs_with_name = {{softmax, "y"}};

    graph.SetInputs(inputs).SetOutputs(outputs);

    return true;
}



void updateSubgraphList(const vector<SubgraphLhq>& src_list, vector<vector<SubgraphLhq> >& dst_list, const GNode& node) {
	
	vector<GNodePtr> pre_nodes; // ��ȡ�ڵ��ȫ������ڵ�
	size_t index_num = node.GetInputsSize(); 
	for (size_t i = 0;i < index_num;i++) {
		std::pair<GNodePtr, int32_t> pre_node = node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
		pre_nodes.push_back(pre_node.first);
	}
	//cout<<"pre_nodes size"<<pre_nodes.size()<<endl;
	
	vector<int> parent_indexes; // ��¼���а�������ڵ����ͼ�±꣬����ͼ����������(data, placehoder, const)�������κ���ͼ��,��������Ҳ����
	for (size_t i = 0;i < src_list.size();i++) {
		for (const auto& pre_node : pre_nodes) {
			if (src_list[i].ContainNode(pre_node)) {
				parent_indexes.push_back(static_cast<int32_t>(i));
				break;
			}
		}
	}
	std::cout<<"parent_indexes size"<<parent_indexes.size()<<endl;
	//std::cout<<"index num"<<index_num<<endl;
	if (parent_indexes.size() > index_num) { // �����ͼ����?
		std:cout<<"One node in multi subgraphs!"<<std::endl;
		return;
	}
	
	// ����Ϊͼ
	vector<SubgraphLhq> new_list(src_list); 
	SubgraphLhq new_graph(node);
	new_list.push_back(new_graph);
	dst_list.push_back(new_list);
	if (parent_indexes.size() == 0) // ���û�и��ڵ㣬���������Լ���һ��ͼ�����������ӽڵ���԰����������ڵ��ں���һ�� 
		return;  
	
	// �뵥�����ڵ�������ͼ�ں�
	for (int a : parent_indexes) {
		new_list = src_list;
		new_list[a].AddNode(node);
		dst_list.push_back(new_list);
	} 
	if (parent_indexes.size() == 1) return; // ֻ��һ������ͼ 
	
	// ���������ڵ�������ͼ�ں�
	for (int i = 0;i < parent_indexes.size();i++) {
	     //cout<<"Merge two parent graph"<<endl;
		for (int j = i + 1;j < parent_indexes.size();j++) {
			new_list = src_list;
			SubgraphLhq& src_a = new_list[parent_indexes[i]];
			SubgraphLhq& src_b = new_list[parent_indexes[j]]; 
			src_a.Merge(src_b);
			new_list.erase(new_list.begin() + parent_indexes[j]); // ɾ�����ںϵ���ͼ
			src_a.AddNode(node); 
			dst_list.push_back(new_list);
		}
	} 
	 
	
}

bool CompareSubgraphList(const vector<SubgraphLhq>& list_a, const vector<SubgraphLhq>& list_b) {
	double pScore_a = 0.0, pScore_b = 0.0;
	for (const auto& sgraph : list_a) {
		pScore_a += sgraph.GetPScore();
	}
	for (const auto& sgraph : list_b) {
		pScore_b += sgraph.GetPScore();
	}
	return pScore_a > pScore_b;
}

void ShowSubgraphSolution(const vector<vector<SubgraphLhq> >& subgraph_alter_space) { // ���������ѡ�����ľ�������?
	int solution_num = subgraph_alter_space.size();
	for (int i = 0;i < solution_num;i++) {
		std::cout<<"-----The number:"<<i<< "subgraph partition scheme------"<<std::endl;
		vector<SubgraphLhq> solution = subgraph_alter_space[i];
		int subgraph_num = solution.size();
		cout<<"--Total subgraph number:"<<subgraph_num<<endl; 
		for (int j = 0;j < subgraph_num;j++) {
			cout<<"--Subgraph number:"<<j<<endl;
			solution[j].ShowNodes();
		}
	}
}

bool SearchGraph(Graph &graph, const int& K = 4) {
    // Option: If you need to know shape and type info of node, you can call infer shape interface:
    // aclgrphInferShapeAndType ,then view this info by graph file which generated by dump graph
    // interface: aclgrphDumpGraph.
    
    // ������data, placeholder, const
    std::cout<<"Search Graph Start."<<std::endl;
    
    Py_Initialize();
	  if (!Py_IsInitialized()) {
		  cout << "python init fail" << endl;
		  return 0;
	  }
	  PyRun_SimpleString("import sys");
	  PyRun_SimpleString("sys.path.append('/home/HwHiAiUser/AscendProjects/searchGraph/scripts')");
	  
    vector<vector<SubgraphLhq> > subgraph_alter_space; // ��ѡ��ͼ�ռ䣬��һά�ȴ���n����ͼ���ַ������ڶ�ά�ȴ��汻�ֳ���m����ͼ 
    
    std::vector<GNode> nodes = graph.GetAllNodes();
    graphStatus ret = GRAPH_FAILED;
    for (auto &node : nodes) {  // Ĭ���������� 
        ge::AscendString name;
        ret = node.GetName(name);
        if (ret != GRAPH_SUCCESS) {
            std::cout<<"Get node name failed."<<std::endl;
            return false;
        }
        std::string node_name(name.GetString());
        std::cout<<"Find node "<<node_name<<std::endl; // ��ӡ������
        
        ge::AscendString type; // ���node_type,�����Ƿ��� 
	      graphStatus ret = node.GetType(type);	
	      if (ret != GRAPH_SUCCESS) {
          std::cout<<"Get node type failed."<<std::endl;
          return false;
        }
        std::string node_type(type.GetString());
        
        const std::unordered_set<std::string> ignore_set = {"Const", "Data", "PlaceHolder"}; // ���������������ӣ�������
        if (ignore_set.find(node_type) != ignore_set.end()) {
            continue;
        }
		 
        if (subgraph_alter_space.empty()) { // Ϊ�յ�������½�һ��?
        	vector<SubgraphLhq> sgraph_list;
        	SubgraphLhq sgraph0(node);
        	sgraph_list.push_back(sgraph0);
        	subgraph_alter_space.push_back(sgraph_list);
        	continue;
	    } 
		
 		vector<vector<SubgraphLhq> > tem_alter_space; // �����ݴ���һ�׶εı�ѡ��ͼ 
        // ������ѡ��ͼ�ռ�
		for (int i = 0;i < subgraph_alter_space.size();i++) {
			/* 
			 * ��ÿ������ڵ㣬�������ڵ���ͼ�����Խ��ýڵ�����?
			 * ���Խ�������ڵ����ڵ���ͼ�ʹ˽ڵ�����һ��?
			 * ���ڴ󲿷����ӵ������������ᳬ��4���������ܹ����Ե��ں����಻�ᳬ�� 17�֡� 
			 */
			updateSubgraphList(subgraph_alter_space[i], tem_alter_space, node);
			cout<<"alter_space size"<<tem_alter_space.size()<<endl;
		} 
		
		// ����pScore���� 
		std::sort(tem_alter_space.begin(), tem_alter_space.end(), CompareSubgraphList);
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
        cout << "[ERROR]input arg num must be 3! " << endl;
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
        std::string tfPath = "../data/tf_test.pb";
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
        std::string caffePath = "../data/caffe_test.prototxt";
        std::string weigtht = "../data/caffe_test.caffemodel";
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
    //SearchGraph(graph1, 3);
    // 2. system init
    std::map<AscendString, AscendString> global_options = {
        {AscendString(ge::ir_option::SOC_VERSION), AscendString(argv[kSocVersion])}  ,
    };
    auto status = aclgrphBuildInitialize(global_options);
    if (status == GRAPH_SUCCESS) {
        cout << "Build Initialize SUCCESS!" << endl;
    }
    else {
        cout << "Build Initialize Failed!" << endl;
    }
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
    status = aclgrphSaveModel("model_with_subgraph", model1);
    if (status == GRAPH_SUCCESS) {
        cout << "Save Offline Model1 SUCCESS!" << endl;
    }
    else {
        cout << "Save Offline Model1 Failed!" << endl;
    }

    // release resource
    aclgrphBuildFinalize();
    return 0;
}
