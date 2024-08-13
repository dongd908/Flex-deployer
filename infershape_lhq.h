#ifndef INFERSHAPELHQ_H
#define INFERSHAPELHQ_H

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

class InferShapeLhq {
public:
    void InferOutputDesc(GNode& node);
private:
    void InferOutputDescConv2D(GNode& node);
};

#endif //INFERSHAPELHQ_H
