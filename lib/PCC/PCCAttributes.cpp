#include "PCC/PCCAttributes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::pcc;
//===----------------------------------------------------------------------===//
// ODS Boilerplate
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "PCC/PCCAttributes.cpp.inc"

void mlir::pcc::PCCDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "PCC/PCCAttributes.cpp.inc"
      >();
}

Attribute PCCDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                     ::mlir::Type type) const {
  StringRef attrName;
  Attribute attr;
  if (parser.parseKeyword(&attrName))
    return {};
  auto parseResult =
      generatedAttributeParser(getContext(), parser, attrName, type, attr);

  if (parseResult.hasValue())
    return attr;

  parser.emitError(parser.getNameLoc(),
                   "Parsing Unexpected Attribute '" + attrName + "'");
  return {};
}

void PCCDialect::printAttribute(::mlir::Attribute attr,
                                ::mlir::DialectAsmPrinter &os) const {
  if (failed(generatedAttributePrinter(attr, os)))
    llvm_unreachable("Unexpected attribute!");
}

//===----------------------------------------------------------------------===//
// StateAttr
//===----------------------------------------------------------------------===//
Attribute StateAttr::parse(::mlir::MLIRContext *context,
                           ::mlir::DialectAsmParser &parser,
                           ::mlir::Type /*type*/) {

  llvm::StringRef state;
  if (parser.parseLess() || parser.parseKeyword(&state) ||
      parser.parseGreater())
    return {};

  return StateAttr::get(StateType::get(context, state.str()));
}

void StateAttr::print(::mlir::DialectAsmPrinter &printer) const {
  printer << StateAttr::getMnemonic() << "<" << getAttrData() << ">";
}