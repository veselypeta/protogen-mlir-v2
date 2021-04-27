#include "PCC/PCCTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <string>

/// Types
/// ::= id
/// Parse a type registered to this dialect.
using mlir::TypeStorageAllocator;
using namespace mlir;
using namespace mlir::pcc;

// Type Printer for PCC Types
// Uses the TypeSwitch Class to help printing types
void PCCType::print(raw_ostream &os) const {
  // Not sure what this does
  // auto printWdithQualifier = [&](Optional<int32_t> width) {
  //     if (width)
  //         os << '<' << width.getValue() << '>';
  // };

  TypeSwitch<PCCType>(*this)
      .Case<IDType>([&](IDType) { os << "id"; })
      .Case<NetworkType>([&](NetworkType netType) {
        os << "network<" << netType.getOrdering() << ">";
      })
      .Default([](Type) { assert(0 && "unkown dialect type to print!"); });
}

static ParseResult parseType(PCCType &result, DialectAsmParser &parser) {
  StringRef name;
  if (parser.parseKeyword(&name))
    return failure();

  MLIRContext *context = parser.getBuilder().getContext();

  if (name.equals("clock")) {
    // odd comma syntax
    return result = IDType::get(context), success();
  } else if (name.equals("network")) {
    std::string order;
    NetworkType::Ordering ordertype;
    // <ordered> //
    if (parser.parseLess() || parser.parseKeyword(order) ||
        parser.parseGreater()) {
      return failure();
    }
    if (order == "ordered") {
      ordertype = NetworkType::Ordering::ORDERED;
    } else if (order == "unordered") {
      ordertype = NetworkType::Ordering::UNORDERED;
    } else {
      return parser.emitError(
                 parser.getNameLoc(),
                 "network type has invalid ordering (can be only ordered "
                 "or unordered) " +
                     order + " is invalid!"),
             failure();
    }

    return result = NetworkType::get(context, ordertype), success();
  }
  return parser.emitError(parser.getNameLoc(), "unknown pcc type"), failure();
}

Type PCCDialect::parseType(::mlir::DialectAsmParser &parser) const {
  PCCType result;
  if (::parseType(result, parser))
    return Type();

  return result;
}

void PCCDialect::printType(::mlir::Type type,
                           ::mlir::DialectAsmPrinter &os) const {
  type.print(os.getStream());
}

//===----------------------------------------------------------------------===//
// Network Type
//===----------------------------------------------------------------------===//
namespace mlir {
namespace pcc {
namespace detail {
struct NetworkTypeStorage : public mlir::TypeStorage {
  using KeyTy = NetworkType::Ordering;
  NetworkTypeStorage(KeyTy value) : value{value} {}

  // compare
  bool operator==(const KeyTy &key) const { return key == value; }

  // define a hash function for key type
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static NetworkTypeStorage *construct(TypeStorageAllocator &allocator,
                                       KeyTy key) {
    return new (allocator.allocate<NetworkTypeStorage>())
        NetworkTypeStorage(key);
  }

  // data held by the storage class
  KeyTy value;
};
} // namespace detail
} // namespace pcc
} // namespace mlir

NetworkType NetworkType::get(MLIRContext *context,
                             NetworkType::Ordering ordering) {
  return Base::get(context, ordering);
}

std::string NetworkType::getOrdering() {
  Ordering order = getImpl()->value;
  if (order == Ordering::UNORDERED) {
    return "unordered";
  }
  return "ordered";
}
