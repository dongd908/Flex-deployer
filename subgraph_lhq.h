#ifndef SUBGRAPHLHQ_H  
#define SUBGRAPHLHQ_H 

#include <memory>
#include <unordered_map>
#include <vector>
#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "attr_value.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_ir_build.h"
#include "all_ops.h"

using namespace std;
using namespace ge;
using ge::Operator;

class SubgraphLhq;
using SubgraphLhqPtr = std::shared_ptr<SubgraphLhq>;

enum class OpTypeLhq: std::uint32_t {  // ignore Data, PlaceHolder, Const node, they are input
	None = 0,
	Conv2D1x1 = 1,
	Conv2D2x2 = 2,
	Conv2D3x3 = 3,
	Conv2D5x5 = 4,   
	Conv2D7x7 = 5,
	MatMul = 6,
	Relu = 7,
	LeakyRelu = 8,
	Sigmoid = 9,
	SoftmaxV2 = 10,
	BiasAdd = 11,
	BatchNorm = 12,
	Pad = 13,
	Slice = 14,
	SplitV = 15,
	Concat = 16,
	Reshape = 17,
	Upsample = 18,
	Add = 19,
	Mul = 20,
	Exp = 21,
	Sub = 22,
	BNInference = 23,
	Scale = 24,
	Eltwise = 25,
	Conv2D = 26
};

const static unordered_map<std::string, OpTypeLhq> cann2feature_optype = {
	{"Conv2D1x1", OpTypeLhq::Conv2D1x1},
	{"Conv2D2x2", OpTypeLhq::Conv2D2x2},
	{"Conv2D3x3", OpTypeLhq::Conv2D3x3},
	{"Conv2D5x5", OpTypeLhq::Conv2D5x5},
	{"Conv2D7x7", OpTypeLhq::Conv2D7x7},
	{"BiasAdd", OpTypeLhq::BiasAdd},
	{"Relu", OpTypeLhq::Relu},
	{"LeakyRelu", OpTypeLhq::LeakyRelu},
	{"MatMul", OpTypeLhq::MatMul},
	{"Mul", OpTypeLhq::Mul},
	{"Add", OpTypeLhq::Add},
	{"Sub", OpTypeLhq::Sub},
	{"SoftmaxV2", OpTypeLhq::SoftmaxV2},
	{"BatchNorm", OpTypeLhq::BatchNorm},
	{"Exp", OpTypeLhq::Exp},
  {"Reshape", OpTypeLhq::Reshape},
  {"Conv2D", OpTypeLhq::Conv2D},
  {"Scale", OpTypeLhq::Scale},
  {"BNInference", OpTypeLhq::BNInference},
  {"Eltwise", OpTypeLhq::Eltwise},
  {"Sigmoid", OpTypeLhq::Sigmoid}
};

enum class InputFormatLhq: std::uint32_t {
	None = 0,
	NCHW = 1,
	NHWC = 2,
	ND = 3,
	HWCN = 4,
	NC1HWC0 = 5,
	FRACTAL_Z = 6
};

const static unordered_map<Format, InputFormatLhq> cann2feature_format = {
  {FORMAT_NCHW, InputFormatLhq::NCHW},
  {FORMAT_NHWC, InputFormatLhq::NHWC},
  {FORMAT_HWCN, InputFormatLhq::HWCN},
  {FORMAT_ND, InputFormatLhq::ND},
  {FORMAT_NC1HWC0, InputFormatLhq::NC1HWC0},
  {FORMAT_FRACTAL_Z, InputFormatLhq::FRACTAL_Z}
};

enum class OpCategoryLhq: std::uint32_t {
	Cube = 86,
	Accu = 87,
	Simple = 88,
	Broadcast = 89,
	Reduce = 90,
	Reshape = 91,
	ConcatASlice = 92,
	Complex = 93
};

const static unordered_map<std::string, OpCategoryLhq> cann2feature_category = {
	{"Conv2D1x1", OpCategoryLhq::Cube},
	{"Conv2D2x2", OpCategoryLhq::Cube},
	{"Conv2D3x3", OpCategoryLhq::Cube},
	{"Conv2D5x5", OpCategoryLhq::Cube},
	{"Conv2D7x7", OpCategoryLhq::Cube},
	{"BiasAdd", OpCategoryLhq::Accu},
	{"Relu", OpCategoryLhq::Accu},
	{"LeakyRelu", OpCategoryLhq::Accu},
	{"MatMul", OpCategoryLhq::Cube},
	{"Mul", OpCategoryLhq::Simple},
	{"Add", OpCategoryLhq::Simple},
	{"Sub", OpCategoryLhq::Simple},
	{"Exp", OpCategoryLhq::Simple},
	{"SoftmaxV2", OpCategoryLhq::Reduce},
	{"Reshape", OpCategoryLhq::Reshape},
	{"BatchNorm", OpCategoryLhq::Accu},
	{"BNInference", OpCategoryLhq::Accu},
  	{"Scale", OpCategoryLhq::Accu},
  	{"Eltwise", OpCategoryLhq::Accu},
  	{"Sigmoid", OpCategoryLhq::Accu}
};

enum class DataTypeLhq: std::uint32_t {
  None = 0,
  FLOAT = 1,
  FLOAT16 = 2,
  INT8 = 3,
  INT16 = 4,
  INT32 = 5,
  INT64 = 6,
  UINT8 = 7,
  UINT16 = 8,
  UINT32 = 9,
  UINT64 = 10
};

const static unordered_map<DataType, DataTypeLhq> cann2feature_datatype = {
  {DT_FLOAT, DataTypeLhq::FLOAT},
  {DT_FLOAT16, DataTypeLhq::FLOAT16},
  {DT_INT8, DataTypeLhq::INT8},
  {DT_INT16, DataTypeLhq::INT16},
  {DT_INT32, DataTypeLhq::INT32},
  {DT_INT64, DataTypeLhq::INT64},
  {DT_UINT8, DataTypeLhq::UINT8},
  {DT_UINT16, DataTypeLhq::UINT16}
};
/*
enum class FusionModeLhq: std::uint32_t {
	None = 5101,
	Static = 5102,
	Dsl = 5103
};
*/
class SubgraphLhq{
public:
	SubgraphLhq(){}
	SubgraphLhq(GNode input0);
	SubgraphLhq(const SubgraphLhq& other); //shallow copy, cause GNode is not change, so it doesn't matter which block you point to
	//~SubgraphLhq();  use default destructor
	
	bool AddNode(const GNode& node); // Add GNode to subgraph. If success, return 1, else return -1
	double UpdatePScore();
	void Merge(const SubgraphLhqPtr& other);
	std::vector<std::pair<GNodePtr, int32_t>>::iterator FindInput(const std::pair<GNodePtr, int32_t>& input);
	std::vector<std::pair<GNodePtr, int32_t>>::iterator FindOutput(const std::pair<GNodePtr, int32_t>& output);
	bool LoopCheck();
	
	bool operator<(const SubgraphLhq& other) const;
	bool operator==(const SubgraphLhq& other) const;
	double GetPScore() const; 
	int GetSize() const;
	bool ContainNode(const GNodePtr& node) const;
	bool ContainOutput(const std::pair<GNodePtr, int32_t>& output) const;
	void ShowNodes() const;

	std::vector<std::weak_ptr<SubgraphLhq> > father_subgraph; // the father subgraph of this
	std::vector<std::weak_ptr<SubgraphLhq> > son_subgraph;
		
private:
        /*
	std::unordered_map<OpTypeLhq, int> op_types;  // the features of subgraph
	int input_number;
	std::vector<TensorDesc> input_desc;
	std::unordered_map<OpCategoryLhq, int> op_categories;
	FusionModeLhq fusion_mode = FusionModeLhq::None;*/
	std::vector<GNodePtr> node_list;  // the nodes in subgraph
	// <node, port_idx>, if output port has multi connection, push multiple times
	std::vector<std::pair<GNodePtr, int32_t>> input_nodes; // (1)in the subgraph (2) need data from nodes out of subgraph which is not constant
	std::vector<std::pair<GNodePtr, int32_t>> output_nodes;// (1)in the subgraph (2) has output node out of subgraph
	std::vector<TensorDesc> input_constant;
	double pScore = 0.0;
	double dpCoff = 0.0;

	string GetNodeType(const GNode& node) const;

	double CalculateDP();
	bool Feature2Libsvm(vector<int>& col, vector<int>& data);
};



#endif
