#include "PCC/PCCOps.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::pcc;

static void print(ProcessOp op, OpAsmPrinter &p) {
  auto procType = op.getType();
  mlir::function_like_impl::printFunctionLikeOp(
      p, op, procType.getInputs(), /*isVariadic=*/false, procType.getResults());
}

static ParseResult parseProcessOp(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes,
                          ArrayRef<Type> results,
                          function_like_impl::VariadicFlag, std::string &) {
    return builder.getFunctionType(argTypes, results);
  };
  return function_like_impl::parseFunctionLikeOp(
      parser, result, /*allowVariadic=*/false, buildFuncType);
}

#define GET_OP_CLASSES
#include "PCC/PCC.cpp.inc"