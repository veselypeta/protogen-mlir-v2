#include "PCC/PCCOps.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::pcc;

// - Process Op
static void print(ProcessOp op, OpAsmPrinter &p) {
  auto procType = op.getType();
  mlir::function_like_impl::printFunctionLikeOp(
      p, op, procType.getInputs(), /*isVariadic=*/false, procType.getResults());
}

static ParseResult parseProcessOp(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes,
                          ArrayRef<Type> results,
                          mlir::function_like_impl::VariadicFlag, std::string &) {
    return builder.getFunctionType(argTypes, results);
  };
  return mlir::function_like_impl::parseFunctionLikeOp(
      parser, result, /*allowVariadic=*/false, buildFuncType);
}

static void print(TransactionOp op, OpAsmPrinter &p) {
  p << op->getName() << ' ';
  Region &body = op->getRegion(0);
  p.printRegion(body);
}

static ParseResult parseTransactionOp(OpAsmParser &parser, OperationState &result){
  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  auto *body = result.addRegion();
  ParseResult parseResult = parser.parseRegion(*body);
  llvm::SMLoc loc = parser.getCurrentLocation();
  if(failed(parseResult))
    return failure();

  if(body->empty())
    return parser.emitError(loc);

  return success();
}

#define GET_OP_CLASSES
#include "PCC/PCC.cpp.inc"