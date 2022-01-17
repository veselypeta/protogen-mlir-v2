#include "FSM/FSMOps.h"
#include "Support/LLVM.h"
#include "mlir/IR/FunctionImplementation.h"
#include <mlir/IR/SymbolTable.h>

using namespace mlir;
using namespace mlir::fsm;

//===----------------------------------------------------------------------===//
// MachineOp
//===----------------------------------------------------------------------===//
void MachineOp::build(::mlir::OpBuilder &odsBuilder,
                      ::mlir::OperationState &odsState, StringRef name,
                      FunctionType type, ArrayRef<NamedAttribute> attrs,
                      ArrayRef<DictionaryAttr> argAttrs) {
  odsState.addAttribute(::mlir::SymbolTable::getSymbolAttrName(),
                        odsBuilder.getStringAttr(name));
  odsState.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  odsState.attributes.append(std::begin(attrs), std::end(attrs));
  odsState.addRegion();
  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_like_impl::addArgAndResultAttrs(odsBuilder, odsState, argAttrs,
                                           /*resultAttrs*/ llvm::None);
}

static ParseResult parseMachineOp(OpAsmParser &parser, OperationState &result) {

  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes,
                          ArrayRef<Type> results,
                          function_like_impl::VariadicFlag, std::string &) {
    return builder.getFunctionType(argTypes, results);
  };

  return function_like_impl::parseFunctionLikeOp(
      parser, result, /*allowVariadic=*/false, buildFuncType);
}

static void print(MachineOp op, OpAsmPrinter &p) {
  FunctionType fnType = op.getType();
  function_like_impl::printFunctionLikeOp(
      p, op, fnType.getInputs(), /*isVariadic=*/false, fnType.getResults());
}

//===----------------------------------------------------------------------===//
// TransitionOp
//===----------------------------------------------------------------------===//
void TransitionOp::build(::mlir::OpBuilder &odsBuilder,
                         ::mlir::OperationState &odsState, StringRef name,
                         FunctionType type, SymbolRefAttr nextState,
                         ArrayRef<NamedAttribute> attrs,
                         ArrayRef<DictionaryAttr> argAttrs) {
  odsState.addAttribute(::mlir::SymbolTable::getSymbolAttrName(),
                        odsBuilder.getStringAttr(name));
  odsState.addAttribute(getTypeAttrName(), TypeAttr::get(type));

  if (nextState)
    odsState.addAttribute(TransitionOp::nextStateAttrName(odsState.name),
                          nextState);

  odsState.attributes.append(std::begin(attrs), std::end(attrs));
  odsState.addRegion();
  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_like_impl::addArgAndResultAttrs(odsBuilder, odsState, argAttrs,
                                           /*resultAttrs*/ llvm::None);
}

static ParseResult parseTransitionOp(OpAsmParser &parser,
                                     OperationState &result) {
  auto buildFunctionType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_like_impl::VariadicFlag, std::string &) -> FunctionType {
    return builder.getFunctionType(argTypes, results);
  };
  return function_like_impl::parseFunctionLikeOp(
      parser, result, /*allowVariadic*/ false, buildFunctionType);
}

static void print(TransitionOp op, OpAsmPrinter &p) {
  FunctionType fnType = op.getType();
  function_like_impl::printFunctionLikeOp(p, op, fnType.getInputs(), false,
                                          fnType.getResults());
}


//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//
void VariableOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), name());
}

//===----------------------------------------------------------------------===//
// UpdateOp
//===----------------------------------------------------------------------===//
VariableOp UpdateOp::getVariable(){
  return variable().getDefiningOp<VariableOp>();
}

static LogicalResult verifyUpdateOp(UpdateOp op){
  if(!op.getVariable())
    return op.emitOpError("destination is not a variable operation");
  return success();
}

//===----------------------------------------------------------------------===//
// StateOp
//===----------------------------------------------------------------------===//
void StateOp::build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, StringRef state) {
  StateOp::build(odsBuilder, odsState, state,
                 /*isTransient*/ nullptr, /*prevTransition*/ nullptr);
}

//===----------------------------------------------------------------------===//
// MessageOp
//===----------------------------------------------------------------------===//
static LogicalResult verifyMessageOp(MessageOp op) {
  if (op.getResult().getType().isa<MsgType>())
    return success();
  return failure();
}

//===----------------------------------------------------------------------===//
// AccessOp
//===----------------------------------------------------------------------===//
static LogicalResult verifyAccessOp(AccessOp op) {
  if (op.msg().getType().isa<MsgType>())
    return success();
  return failure();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "FSM/FSM.cpp.inc"
