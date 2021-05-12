#include "PCC/PCCOps.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::pcc;

static void print(ProcessOp op, OpAsmPrinter &p) {
  ProcessType procType = op.getProcType();
  impl::printFunctionLikeOp(p, op, procType.getInputs(), /*isVariadic=*/false,
                            procType.getResults());
}


#define GET_OP_CLASSES
#include "PCC/PCC.cpp.inc"