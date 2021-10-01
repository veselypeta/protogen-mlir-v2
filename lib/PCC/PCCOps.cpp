#include "PCC/PCCOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include <mlir/IR/SymbolTable.h>

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
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         mlir::function_like_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };
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
  function_like_impl::addArgAndResultAttrs(builder, state, argAttrs,
                                           llvm::None);
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
// TransitionOp
//===----------------------------------------------------------------------===//

// Transition Op looks something like this:
/*
 * pcc.when [GetM] (%arg0: pcc.struct<>){
 *  ....
 * }
 */

static void print(TransitionOp transitionOp, OpAsmPrinter &p) {
  p << mlir::pcc::TransitionOp::getOperationName() << ' ';
  p << "[ " << transitionOp.guard().str() << " ]" << ' ';
  p << '(';
  p.printRegionArgument(transitionOp.body().getArgument(0));
  p << ')';
  p.printRegion(transitionOp.body(), /*printEntryBlockArgs*/ false,
                /*printBlockTerminators*/ true);
}

static ParseResult parseTransitionOp(OpAsmParser &parser,
                                     OperationState &result) {
  auto &builder = parser.getBuilder();
  llvm::StringRef guardMsgName;
  if (parser.parseLSquare() || parser.parseKeyword(&guardMsgName) ||
      parser.parseRSquare()) {
    return mlir::failure();
  }

  std::string guardName = guardMsgName.str();
  result.addAttribute("guard", builder.getStringAttr(guardName));

  // parse the signature

  OpAsmParser::OperandType argument;
  Type argumentType;

  if (parser.parseLParen() || parser.parseRegionArgument(argument) ||
      parser.parseColonType(argumentType) || parser.parseRParen())
    return failure();

  result.addAttribute("msgType", TypeAttr::get(argumentType));

  // parse the body
  auto *body = result.addRegion();
  ParseResult parseResult = parser.parseRegion(*body, argument, argumentType,
                                               /*enableNameShadowing*/ false);

  if (failed(parseResult))
    return failure();

  return success();
}

#define GET_OP_CLASSES
#include "PCC/PCC.cpp.inc"