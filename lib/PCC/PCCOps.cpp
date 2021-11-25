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

bool ProcessOp::hasNestedAwait() {
  return false;   // TODO - implement this method properly
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

//===----------------------------------------------------------------------===//
// StateUpdateOp
//===----------------------------------------------------------------------===//

// State Update Op looks something like
/*
 * pcc.update [ cl ] %cache %value
 */

static void print(StateUpdateOp updateOp, OpAsmPrinter &p) {
  p << updateOp->getName() << ' ';
  p << '[' << updateOp.field() << ']' << ' ';
  p.printOperand(updateOp.mach());
  p << ':';
  p.printType(updateOp.mach().getType());
  p << ' ';
  p.printOperand(updateOp.value());
  p << ':';
  p.printType(updateOp.value().getType());
}

static ParseResult parseStateUpdateOp(OpAsmParser &parser,
                                      OperationState &result) {
  auto &builder = parser.getBuilder();
  llvm::StringRef fieldRef;
  if (parser.parseLSquare() || parser.parseKeyword(&fieldRef) ||
      parser.parseRSquare())
    return failure();

  result.addAttribute("field", builder.getStringAttr(fieldRef));
  OpAsmParser::OperandType mach;
  StructType macType;
  if (parser.parseOperand(mach) || parser.parseColonType(macType))
    return failure();
  if (parser.resolveOperand(mach, macType, result.operands))
    return mlir::failure();

  OpAsmParser::OperandType value;
  Type valueType;
  if (parser.parseOperand(value) || parser.parseColonType(valueType))
    return failure();
  if (parser.resolveOperand(value, valueType, result.operands))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// MsgSendOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

/*
 * pcc.if %cond {
 *    ... then operations ...
 * } else {
 *    ... else operations ...
 * }
 */

static void print(mlir::pcc::IfOp op, OpAsmPrinter &p) {
  bool printBlockTerminators = false;
  bool printEntryBlockArgs = false;
  p << mlir::pcc::IfOp::getOperationName() << ' ';

  p.printOperand(op.condition());
  p.printRegion(op.thenRegion(), printEntryBlockArgs, printBlockTerminators);

  auto &elseRegion = op.elseRegion();
  if (!elseRegion.empty()) {
    p << " else";
    p.printRegion(op.elseRegion(), printEntryBlockArgs, printBlockTerminators);
  }
  p.printOptionalAttrDict(op->getAttrs());
}

static ParseResult parseIfOp(OpAsmParser &parser, OperationState &result) {
  // create the regions for 'then'
  result.regions.reserve(2);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();

  OpAsmParser::OperandType cond;
  Type i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, i1Type, result.operands))
    return failure();
  if (parser.parseRegion(*thenRegion, /*arguments=*/{}, /*argTypes*/ {}))
    return failure();

  if (!parser.parseOptionalKeyword("else")) {
    if (parser.parseRegion(*elseRegion, {}, {}))
      return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void IfOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &result,
                 Value cond, bool withElseRegion) {
  result.addOperands(cond);

  OpBuilder::InsertionGuard guard(builder);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();

  builder.createBlock(thenRegion);
  if (withElseRegion)
    builder.createBlock(elseRegion);
}

Block *IfOp::thenBlock() { return &thenRegion().back(); }

Block *IfOp::elseBlock() {
  Region &r = elseRegion();
  if (r.empty())
    return nullptr;
  return &r.back();
}

//===----------------------------------------------------------------------===//
// InlineConstOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, InlineConstOp &op) {
  p << mlir::pcc::InlineConstOp::getOperationName() << ' ';
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs*/ {"value"});
  if (op->getAttrs().size() > 1)
    p << ' ';
  p << op.value();
}

static ParseResult parseInlineConstOp(OpAsmParser &parser,
                                      OperationState &result) {
  auto &builder = parser.getBuilder();
  Attribute valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(valueAttr, "value", result.attributes))
    return failure();
  result.addTypes(valueAttr.getType());
  return success();
}

void InlineConstOp::build(::mlir::OpBuilder &odsBuilder,
                          ::mlir::OperationState &odsState, Attribute value) {
  odsState.addAttribute("value", value);
  odsState.addTypes(value.getType());
}

void InlineConstOp::build(::mlir::OpBuilder &odsBuilder,
                          ::mlir::OperationState &odsState, StateType value) {
  auto attr = StateAttr::get(value);
  build(odsBuilder, odsState, attr);
}
#define GET_OP_CLASSES
#include "PCC/PCC.cpp.inc"