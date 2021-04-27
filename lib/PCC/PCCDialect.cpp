#include "PCC/PCCOps.h"
#include <models/Expr.h>

using namespace mlir;
using namespace mlir::pcc;

void PCCDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "PCC/PCC.cpp.inc"
      >();
}