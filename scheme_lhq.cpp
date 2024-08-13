#include <iostream>
#include <fstream>
#include <algorithm>
#include <memory>
#include <string.h>
#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "attr_value.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_ir_build.h"
#include "all_ops.h"

#include <vector>
#include <unordered_set>
#include "subgraph_lhq.h"
#include "scheme_lhq.h"

using namespace std;
using namespace ge;
using ge::Operator;

SchemeLhq::SchemeLhq(const SchemeLhq& other) {
    for (int i =0;i < other.subgraph_list.size();i++) {
        this->subgraph_list.push_back(std::make_shared<SubgraphLhq>(*(other.subgraph_list[i])));
    }
}

SchemeLhq::SchemeLhq(const vector<SubgraphLhqPtr>& subgraph_list) {
    for (int i =0;i < subgraph_list.size();i++) {
        this->subgraph_list.push_back(std::make_shared<SubgraphLhq>(*subgraph_list[i]));
    }
}

void SchemeLhq::AddSubgraph(SubgraphLhqPtr g) {
    this->subgraph_list.push_back(g);
}

void SchemeLhq::RemoveSubgraph(const int idx) {
    this->subgraph_list[idx].reset();
    this->subgraph_list.erase(this->subgraph_list.begin() + idx);
}

int SchemeLhq::GetSize() const {
    return this->subgraph_list.size();
}

SubgraphLhqPtr SchemeLhq::GetSubgraph(int idx) const {
    return this->subgraph_list[idx];
}

double SchemeLhq::GetPScore() const {
    double score = 0.0;
    for (auto x : this->subgraph_list) {
        score += x->GetPScore();
    }
    return score;
}

void SchemeLhq::ShowSubgraph() const {
    int subgraph_num = this->subgraph_list.size();
    cout<<"--Total subgraph number:"<<subgraph_num<<endl;
    for (int j = 0;j < subgraph_num;j++) {
        cout<<"--Subgraph number:"<<j<<endl;
        this->subgraph_list[j]->ShowNodes();

    }
}

bool SchemeLhq::AddNode(const GNode& node, vector<SchemeLhq>& new_list) const {
    graphStatus ret = GRAPH_FAILED;
    vector<std::pair<GNodePtr, int32_t>> pre_nodes; // find parents node
    size_t index_num = node.GetInputsSize();
    for (size_t i = 0;i < index_num;i++) {
        std::pair<GNodePtr, int32_t> pre_node = node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
        ge::AscendString type;  // ignore const & input type
        ret = pre_node.first->GetType(type);
        std::string node_type(type.GetString());
        const std::unordered_set<std::string> ignore_set = {"Const", "Data", "PlaceHolder"}; // these nodes not belong to any subgraph
        if (ignore_set.find(node_type) == ignore_set.end()) {
            pre_nodes.push_back(pre_node);
        }
    }
    //cout<<"pre_nodes size"<<pre_nodes.size()<<endl;
    //cout<<"subgraph_list.size()"<<this->subgraph_list.size()<<endl;
    vector<int> parent_indexes; // find parents subgraph
    for (const auto& pre_node : pre_nodes) {
        for (int i = this->subgraph_list.size() - 1;i >= 0;i--) { // Reverse order is easier to find
            if (this->subgraph_list[i]->ContainOutput(pre_node)) {
                if (!std::count(parent_indexes.begin(), parent_indexes.end(), static_cast<int32_t>(i))) { // one father subgraph counts one time
                    parent_indexes.push_back(static_cast<int32_t>(i));
                }
                break;
            }
        }
    }
    //std::cout<<"parent_indexes size"<<parent_indexes.size()<<endl;

    // independent living
    {
        SchemeLhq new_scheme(this->subgraph_list);
        SubgraphLhqPtr new_graph = std::make_shared<SubgraphLhq>(node);
        new_scheme.AddSubgraph(new_graph);
        new_list.push_back(new_scheme);
    }
    if (parent_indexes.size() == 0) return true;

    // single parent
    for (int idx : parent_indexes) {
        SchemeLhq new_scheme(this->subgraph_list);
        bool re_addnode = new_scheme.GetSubgraph(idx)->AddNode(node);
        if (re_addnode && new_scheme.GetSubgraph(idx)->GetPScore() != 0) { // Discard the subgraph with zero pscore, consider that all subgraphs prefixed with it score 0
            new_list.push_back(new_scheme);
        }     
    }
    if (parent_indexes.size() == 1) return true;
    
    // a family of three
    for (int i = 0;i < parent_indexes.size();i++) {
        //cout<<"Merge two parent graph"<<endl;
        for (int j = i + 1;j < parent_indexes.size();j++) {
            SchemeLhq new_scheme(this->subgraph_list);
            SubgraphLhqPtr father_a = new_scheme.GetSubgraph(parent_indexes[i]);
            SubgraphLhqPtr father_b = new_scheme.GetSubgraph(parent_indexes[j]);
            father_a->Merge(father_b);
            new_scheme.RemoveSubgraph(parent_indexes[j]); // remove second graph
            bool re_addnode = father_a->AddNode(node);
            if (re_addnode && father_a->GetPScore() != 0) {
                new_list.push_back(new_scheme);
            }	
        }
    }
    // TODO check cycle(using son and father subgraph)
    return true;
}
