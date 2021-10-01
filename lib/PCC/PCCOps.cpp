#include "PCC/PCCOps.h"
#include <mlir/IR/SymbolTable.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::pcc;

//===----------------------------------------------------------------------===//
// ProcessOp
//===----------------------------------------------------------------------===//

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

void ProcessOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                      FunctionType type, ArrayRef<NamedAttribute> attrs,
                      ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_like_impl::addArgAndResultAttrs(builder, state, argAttrs, llvm::None);
}

//===----------------------------------------------------------------------===//
// CacheDeclOp
//===----------------------------------------------------------------------===//
void CacheDeclOp::build(::mlir::OpBuilder &odsBuilder,
                        ::mlir::OperationState &odsState, StringRef cacheId,
                        PCCType type, ArrayRef<NamedAttribute> attrs) {
  // FIXME - the use of "id" here is dangerous since if changed id TableGen will
  // break here
  odsState.addAttribute("id", odsBuilder.getStringAttr(cacheId));
  odsState.addTypes(type);
  odsState.attributes.append(attrs.begin(), attrs.end());
}

//===----------------------------------------------------------------------===//
// DirectoryDeclOp
//===----------------------------------------------------------------------===//
void DirectoryDeclOp::build(::mlir::OpBuilder &odsBuilder,
                            ::mlir::OperationState &odsState, StringRef dirId,
                            PCCType type, ArrayRef<NamedAttribute> attrs) {
  // FIXME - the use of "id" here is dangerous since if changed id TableGen will
  // break here
  odsState.addAttribute("id", odsBuilder.getStringAttr(dirId));
  odsState.addTypes(type);
  odsState.attributes.append(attrs.begin(), attrs.end());
}

//===----------------------------------------------------------------------===//
// MessageDeclOp
//===----------------------------------------------------------------------===//
void MsgDeclOp::build(::mlir::OpBuilder &odsBuilder,
                      ::mlir::OperationState &odsState, ::llvm::StringRef mtype,
                      StructType msgMlirType,
                      ArrayRef<mlir::NamedAttribute> attrs) {
  // FIXME - the use of "id" here is dangerous since if changed in TableGen will
  // break here
  odsState.addAttribute("id", odsBuilder.getStringAttr(mtype));
  odsState.addTypes(msgMlirType);
  odsState.attributes.append(attrs.begin(), attrs.end());
}

//===----------------------------------------------------------------------===//
// TransactionOp
//===----------------------------------------------------------------------===//

//void TransactionOp::build(OpBuilder &builder, OperationState &state) {
//
//  Region *reg = state.addRegion();
//  builder.createBlock(reg);
//
//}

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