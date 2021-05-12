#include "PCC/PCCOps.h"

using namespace mlir;
using namespace mlir::pcc;

void PCCDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "PCC/PCC.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ProcessOp
//===----------------------------------------------------------------------===//

ProcessOp ProcessOp::create(Location location, ProcessType type) {
  OperationState state(location, "proc");
  OpBuilder builder(location->getContext());
  ProcessOp::build(builder, state, "no-name-assigned",
                   type);
  return llvm::cast<ProcessOp>(Operation::create(state));
}

ProcessType ProcessOp::getProcType() {
  return getTypeAttr().getValue().cast<ProcessType>();
}
