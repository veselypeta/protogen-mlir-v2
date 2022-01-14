#include "FSM/FSMTypes.h"
#include "Support/LLVM.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"


#define GET_TYPEDEF_CLASSES
#include "FSM/FSMTypes.cpp.inc"

using namespace mlir;
using namespace mlir::fsm;

//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//
void FSMType::print(llvm::raw_ostream &os) const {
  TypeSwitch<FSMType>(*this)
      .Case<IDType>([&](IDType) { os << "id"; })
      .Case<DataType>([&](DataType) { os << "data"; })
      .Case<MsgType>([&](MsgType) { os << "msg"; })
      .Default([](Type) { assert(0 && "unknown fsm dialect type"); });
}

void FSMDialect::printType(Type type, DialectAsmPrinter &os) const {
  type.cast<FSMType>().print(os.getStream());
}

//===----------------------------------------------------------------------===//
// Type Parsing
//===----------------------------------------------------------------------===//
ParseResult parseType(FSMType &result, DialectAsmParser &parser) {
  llvm::StringRef name;
  if (parser.parseKeyword(&name))
    return failure();
  MLIRContext *ctx = parser.getBuilder().getContext();
  if(name.equals("id"))
    return result = IDType::get(ctx), success();
  if(name.equals("data"))
    return result = DataType::get(ctx), success();
  if(name.equals("msg"))
    return result = MsgType::get(ctx), success();

  return parser.emitError(parser.getNameLoc(), "unknown fsm type"), failure();
}

Type FSMDialect::parseType(::mlir::DialectAsmParser &parser) const {
  FSMType result;
  if (::parseType(result, parser))
    return Type{};
  return result;
}

void FSMDialect::registerTypes() {
  addTypes<IDType, DataType, MsgType>();
}