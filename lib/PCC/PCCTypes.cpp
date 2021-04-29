#include "PCC/PCCTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <iostream>
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
        os << ", " << setType.getNumElements() << ">";
      })
      .Case<StructType>([&](StructType structType) {
        os << "struct<";
        llvm::interleaveComma(structType.getElementTypes(), os);
        os << ">";
      })
      .Case<IntRangeType>([&](IntRangeType &intRangeType){
        os << "int_range<" << intRangeType.getStartRange()
           << ", " << intRangeType.getEndRange() << ">";
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
  }
  // END - ID Type Parsing
  else if (name.equals("network")) {
    llvm::StringRef order;
    NetworkType::Ordering ordertype;
    // <ordered> //
    if (parser.parseLess() || parser.parseKeyword(&order) ||
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
  // END - network Type Parsing
  else if (name.equals("state")) {
    llvm::StringRef stateValue;
    if (parser.parseLess() || parser.parseKeyword(&stateValue) ||
        parser.parseGreater()) {
      return failure();
    }

    std::string stateValueStr(stateValue.data());
    // HACK! - for some reason > is parsed into the stateValue StrRef - here i'm
    // removing it
    if (!stateValueStr.empty()) {
      stateValueStr.erase(std::prev(stateValueStr.end()));
    }
    return result = StateType::get(context, stateValueStr), success();
  }
  // END - State Type Parsing
  else if (name.equals("set")) {
    PCCType elementType;
    size_t count = 0;
    if (parser.parseLess() || parser.parseType(elementType) ||
        parser.parseComma() || parser.parseInteger(count) ||
        parser.parseGreater()) {
      return failure();
    }
    return result = SetType::get(elementType, count), success();
  }
  // END - set Type Parsing
  else if (name.equals("struct")) {
    // parese `<`
    if (parser.parseLess())
      return failure();

    SmallVector<mlir::Type, 1> elementTypes;
    do {
      // parse the current element type
      mlir::Type elementType;
      if (parser.parseType(elementType))
        return failure();

      // check what types can be in struct type -- todo

      // add the parsed type to the vector
      elementTypes.push_back(elementType);
      // parse the optional `,`
    } while (succeeded(parser.parseOptionalComma()));
    {
      // parse '>'
      if (parser.parseGreater())
        return failure();
      return result = StructType::get(elementTypes), success();
    }
  }
  // END - struct Type Parsing
  else if(name.equals("int_range")) {
    size_t startRange, endRange;
    if(parser.parseLess() || parser.parseInteger(startRange)
        || parser.parseOptionalComma() || parser.parseInteger(endRange)
        || parser.parseGreater())
      return failure();
    return result = IntRangeType::get(context, startRange, endRange), success();
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

NetworkType::Ordering NetworkType::convertToOrder(llvm::StringRef order){
  if(order == "ordered"){
    return Ordering::ORDERED;
  } else if (order == "unordered") {
    return Ordering::UNORDERED;
  }
  assert(0 && "invalid ordering string was provided");
  return Ordering::ORDERED;
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

//===----------------------------------------------------------------------===//
// Struct Type
//===----------------------------------------------------------------------===//
namespace mlir {
namespace pcc {
namespace detail {
// same as in Toy example
struct StructTypeStorage : public mlir::TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage. For our struct type, we will unique each instance structurally on
  /// the elements that it contains.
  using KeyTy = llvm::ArrayRef<mlir::Type>;

  /// A constructor for the type storage instance.
  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
      : elementTypes(elementTypes) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  /// Define a hash function for the key type. This is used when uniquing
  /// instances of the storage, see the `StructType::get` method.
  /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
  /// have hash functions available, so we could just omit this entirely.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  /// Define a construction function for the key type from a set of parameters.
  /// These parameters will be provided when constructing the storage instance
  /// itself.
  /// Note: This method isn't necessary because KeyTy can be directly
  /// constructed with the given parameters.
  static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
    return KeyTy(elementTypes);
  }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
  /// allocations used to create the type storage and its internal.
  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    // Copy the elements from the provided `KeyTy` into the allocator.
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

    // Allocate the storage instance and construct it.
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(elementTypes);
  }

  KeyTy elementTypes;
};
} // namespace detail
} // namespace pcc
} // namespace mlir

StructType StructType::get(llvm::ArrayRef<mlir::Type> elementTypes) {
  assert(!elementTypes.empty() && "expected at least 1 element type");

  // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  // of this type. The first parameter is the context to unique in. The
  // parameters after the context are forwarded to the storage instance.
  mlir::MLIRContext *ctx = elementTypes.front().getContext();
  return Base::get(ctx, elementTypes);
}

llvm::ArrayRef<mlir::Type> StructType::getElementTypes() {
  return getImpl()->elementTypes;
}

//===----------------------------------------------------------------------===//
// IntRange Type
//===----------------------------------------------------------------------===//

namespace mlir{
namespace pcc{
namespace detail{
struct IntRangeTypeStorage : public mlir::TypeStorage{
  using KeyTy = std::pair<size_t, size_t>;

  IntRangeTypeStorage(size_t startRange, size_t endRange)
      :value{std::make_pair(startRange, endRange)}{}

  bool operator==(const KeyTy &key) const{
    return key == value;
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static IntRangeTypeStorage *construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key){
      return new (allocator.allocate<IntRangeTypeStorage>()) IntRangeTypeStorage(key.first, key.second);
  }

  KeyTy value;
};
}
}
}

IntRangeType IntRangeType::get(MLIRContext *context, size_t startRange, size_t endRange) {
  return Base::get(context, startRange, endRange);
}

size_t IntRangeType::getStartRange() {
  return getImpl()->value.first;
}
size_t IntRangeType::getEndRange() {
  return getImpl()->value.second;
}

// Register Newly Created types to the dialect

void PCCDialect::registerTypes() {
  addTypes<IDType, NetworkType, StateType, SetType, StructType, IntRangeType>();
}