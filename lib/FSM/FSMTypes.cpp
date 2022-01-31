#include "FSM/FSMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "FSM/FSMTypes.cpp.inc"

using mlir::TypeStorageAllocator;
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
      .Case<StateType>([&](StateType) { os << "state"; })
      .Case<RangeType>([&](RangeType t) {
        os << "range<" << t.getStart() << ", " << t.getEnd() << ">";
      })
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
  if (name.equals("id"))
    return result = IDType::get(ctx), success();
  if (name.equals("data"))
    return result = DataType::get(ctx), success();
  if (name.equals("msg"))
    return result = MsgType::get(ctx), success();
  if (name.equals("state"))
    return result = StateType::get(ctx), success();
  if (name.equals("range")) {
    size_t start, end;
    if (parser.parseLess() || parser.parseInteger(start) ||
        parser.parseComma() || parser.parseInteger(end) ||
        parser.parseGreater())
      return failure();

    return result = RangeType::get(ctx, start, end), success();
  }

  return parser.emitError(parser.getNameLoc(), "unknown fsm type"), failure();
}

Type FSMDialect::parseType(::mlir::DialectAsmParser &parser) const {
  FSMType result;
  if (::parseType(result, parser))
    return Type{};
  return result;
}

void FSMDialect::registerTypes() {
  addTypes<IDType, DataType, MsgType, StateType, RangeType>();
}

namespace mlir {
namespace fsm {
namespace detail {
struct RangeTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::pair<size_t, size_t>;

  RangeTypeStorage(size_t startRange, size_t endRange)
      : value{std::make_pair(startRange, endRange)} {}

  bool operator==(const KeyTy &key) const { return key == value; }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static RangeTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<RangeTypeStorage>())
        RangeTypeStorage(key.first, key.second);
  }

  KeyTy value;
};
} // namespace detail
} // namespace fsm
} // namespace mlir
RangeType RangeType::get(MLIRContext *context, size_t startRange,
                         size_t endRange) {
  return Base::get(context, startRange, endRange);
}

size_t RangeType::getStart() { return getImpl()->value.first; }
size_t RangeType::getEnd() { return getImpl()->value.second; }


// verification
LogicalResult mlir::fsm::areTypesCompatible(Type t1, Type t2){
  // Range is compatible with Integer Types
  if((t1.isa<RangeType>() && t2.isa<IntegerType>()) || (t1.isa<IntegerType>() && t2.isa<RangeType>()))
    return success();
  return failure();
}