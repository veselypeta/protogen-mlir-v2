#include "PCC/PCCOps.h"

using namespace mlir;
using namespace mlir::pcc;

void PCCDialect::initialize() {
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "PCC/PCC.cpp.inc"
      >();
}

#include "PCC/PCCDialect.cpp.inc"