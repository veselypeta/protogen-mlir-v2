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

//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

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
      .Case<StateType>([&](StateType stateType) {
        os << "state<" << stateType.getState() << ">";
      })
      .Case<SetType>([&](SetType setType) {
        os << "set<";
        setType.getElementType().print(os);
        os << "," << setType.getNumElements() << ">";
      })
      .Default([](Type) { assert(0 && "unkown dialect type to print!"); });
}

void PCCDialect::printType(::mlir::Type type,
                           ::mlir::DialectAsmPrinter &os) const {
  type.cast<PCCType>().print(os.getStream());

}

//===----------------------------------------------------------------------===//
// Type Parsing
//===----------------------------------------------------------------------===//

static ParseResult parseType(PCCType &result, DialectAsmParser &parser) {
  StringRef name;
  if (parser.parseKeyword(&name))
    return failure();

  MLIRContext *context = parser.getBuilder().getContext();

  if (name.equals("id")) {
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
    } else if (name.equals("state")) {
      std::string stateValue;
      if (parser.parseLess() || parser.parseKeyword(stateValue) ||
          parser.parseGreater()) {
        return failure();
      }
      return result = StateType::get(context, stateValue), success();
    } else if (name.equals("set")) {
      PCCType elementType;
      size_t count = 0;
      if (parser.parseLess() || parser.parseType(elementType) ||
          parser.parseComma() || parser.parseInteger(count) ||
          parser.parseGreater()) {
        return failure();
      }
      return result = SetType::get(elementType, count), success();
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

//===----------------------------------------------------------------------===//
// State Type
//===----------------------------------------------------------------------===//
namespace mlir {
namespace pcc {
namespace detail {
struct StateTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::string;

  StateTypeStorage(KeyTy value) : value{value} {}

  // for comparison operations
  bool operator==(const KeyTy &key) const { return key == value; }
  // define a hash function for key type
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static StateTypeStorage *construct(TypeStorageAllocator &allocator,
                                     KeyTy key) {
    return new (allocator.allocate<StateTypeStorage>()) StateTypeStorage(key);
  }

  KeyTy value;
};
} // namespace detail
} // namespace pcc
} // namespace mlir

std::string StateType::getState() { return getImpl()->value; }

StateType StateType::get(MLIRContext *context, std::string state) {
  return Base::get(context, state);
}

//===----------------------------------------------------------------------===//
// Set Type
//===----------------------------------------------------------------------===//

namespace mlir {
namespace pcc {
namespace detail {
// Set Type Storage holds they PCCType in the set - and an int for the number of
// elements in the set
struct SetTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::pair<PCCType, size_t>;

  SetTypeStorage(KeyTy value) : value{value} {}

  // for comparison operations
  bool operator==(const KeyTy &key) const { return key == value; }

  static SetTypeStorage *construct(TypeStorageAllocator &allocator, KeyTy key) {
    return new (allocator.allocate<SetTypeStorage>()) SetTypeStorage(key);
  }

  // define a hash function for key type
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  // Class holds the value
  KeyTy value;
};
} // namespace detail
} // namespace pcc
} // namespace mlir

SetType SetType::get(PCCType type, size_t count) {
  return Base::get(type.getContext(), std::make_pair(type, count));
}

PCCType SetType::getElementType() { return getImpl()->value.first; }

size_t SetType::getNumElements() { return getImpl()->value.second; }


// Register Newly Created types to the dialect

void PCCDialect::registerTypes() {
  addTypes<IDType, NetworkType, StateType, SetType>();
}