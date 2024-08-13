#ifndef SCHEMELHQ_H
#define SCHEMELHQ_H

#include <vector>
#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "attr_value.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_ir_build.h"
#include "all_ops.h"
#include "subgraph_lhq.h"

using namespace std;
using namespace ge;
using ge::Operator;

class SchemeLhq {
public:
    SchemeLhq(){}
    SchemeLhq(const SchemeLhq& other); // deep copy
    SchemeLhq(const vector<SubgraphLhqPtr>& subgraph_list);

    void AddSubgraph(SubgraphLhqPtr g);
    void RemoveSubgraph(const int idx);

    bool AddNode(const GNode& node, vector<SchemeLhq>& new_list) const;
    int GetSize() const;
    SubgraphLhqPtr GetSubgraph(int idx) const;
    double GetPScore() const;
    void ShowSubgraph() const;

private:
    vector<SubgraphLhqPtr> subgraph_list;
};

#endif

//searchGraph_scheme_lhq_H
