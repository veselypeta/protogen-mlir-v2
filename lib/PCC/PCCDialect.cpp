#include "PCC/PCCOps.h"
#include <mlir/IR/SymbolTable.h>

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
  SmallString<8> argAttrName;
  for (unsigned i = 0, e = type.getNumInputs(); i != e; ++i)
    if (DictionaryAttr argDict = argAttrs[i])
      state.addAttribute(getArgAttrName(i, argAttrName), argDict);
}

// ProcessOp ProcessOp::create(Location location, ProcessType type) {
//   OperationState state(location, "proc");
//   OpBuilder builder(location->getContext());
//   ProcessOp::build(builder, state, "no-name-assigned",
//                    type);
//   return llvm::cast<ProcessOp>(Operation::create(state));
// }

//===----------------------------------------------------------------------===//
// CacheDeclOp
//===----------------------------------------------------------------------===//
void CacheDeclOp::build(::mlir::OpBuilder &odsBuilder,
                        ::mlir::OperationState &odsState, StringRef cacheId,
                        PCCType type, ArrayRef<NamedAttribute> attrs) {
  // FIXME - the use of "id" here is dangerous since if changed id TableGen will break here
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
  // FIXME - the use of "id" here is dangerous since if changed id TableGen will break here
  odsState.addAttribute("id", odsBuilder.getStringAttr(dirId));
  odsState.addTypes(type);
  odsState.attributes.append(attrs.begin(), attrs.end());
}