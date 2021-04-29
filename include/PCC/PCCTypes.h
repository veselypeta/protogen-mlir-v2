#pragma once
#include "PCC/PCCDialect.h"
#include "mlir/IR/Types.h"
#include <string>

namespace mlir {

namespace pcc {
namespace detail {
struct NetworkTypeStorage;
struct SetTypeStorage;
struct StateTypeStorage;
struct StructTypeStorage;
struct IntRangeTypeStorage;
} // namespace detail

// Types
class IDType;
class DataType;
class NetworkType;
class SetType;
class StateType;
class StructType;
class IntRangeType;

//===----------------------------------------------------------------------===//
// PCC Type
//===----------------------------------------------------------------------===//
// --- this is a common base for all PCC Types
class PCCType : public Type {
public:
  void print(raw_ostream &os) const;

  // Support method to enable LLVM-style type casting.
  static bool classof(Type type) {
    return llvm::isa<mlir::pcc::PCCDialect>(type.getDialect());
  }

protected:
  using Type::Type;
};

//===----------------------------------------------------------------------===//
// ID Type
//===----------------------------------------------------------------------===//
/// ID Type inherits from common base class PCCType
class IDType : public PCCType::TypeBase<IDType, PCCType, DefaultTypeStorage> {
public:
  using Base::Base;
  static IDType get(MLIRContext *context) { return Base::get(context); }
};

//===----------------------------------------------------------------------===//
// IntRange Type
//===----------------------------------------------------------------------===//
class DataType
    : public PCCType::TypeBase<DataType, PCCType, DefaultTypeStorage> {
public:
  using Base::Base;
  static DataType get(MLIRContext *context) { return Base::get(context); }
};

//===----------------------------------------------------------------------===//
// State Type
//===----------------------------------------------------------------------===//
class StateType
    : public PCCType::TypeBase<StateType, PCCType, detail::StateTypeStorage> {
public:
  using Base::Base;
  std::string getState();
  static StateType get(MLIRContext *context, std::string state);
};

//===----------------------------------------------------------------------===//
// Network Type
//===----------------------------------------------------------------------===//

// Network Type inherits from common base class PCCType
class NetworkType : public PCCType::TypeBase<NetworkType, PCCType,
                                             detail::NetworkTypeStorage> {
public:
  using Base::Base;
  std::string getOrdering();
  typedef enum { ORDERED, UNORDERED } Ordering;
  static NetworkType get(MLIRContext *context, Ordering order);
  static Ordering convertToOrder(llvm::StringRef order);
};

//===----------------------------------------------------------------------===//
// Set Type
//===----------------------------------------------------------------------===//

class SetType
    : public PCCType::TypeBase<SetType, PCCType, detail::SetTypeStorage> {
public:
  using Base::Base;
  PCCType getElementType();
  size_t getNumElements();
  static SetType get(PCCType elemType, size_t count);
};

//===----------------------------------------------------------------------===//
// Struct Type
//===----------------------------------------------------------------------===//

class StructType
    : public PCCType::TypeBase<StructType, PCCType, detail::StructTypeStorage> {
public:
  using Base::Base;

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be atleast one element type.
  static StructType get(llvm::ArrayRef<mlir::Type> elemTypes);

  /// Returns the element types of this struct type.
  llvm::ArrayRef<mlir::Type> getElementTypes();

  /// Returns the number of element type held by this struct.
  size_t getNumElementTypes() { return getElementTypes().size(); }
};

} // namespace pcc
} // namespace mlir

//===----------------------------------------------------------------------===//
// IntRange Type
//===----------------------------------------------------------------------===//
namespace mlir {
namespace pcc {
class IntRangeType : public PCCType::TypeBase<IntRangeType, PCCType,
                                              detail::IntRangeTypeStorage> {
public:
  using Base::Base;
  static IntRangeType get(MLIRContext *context, size_t startRange,
                          size_t endRange);
  size_t getStartRange();
  size_t getEndRange();
};
} // namespace pcc
} // namespace mlir

namespace llvm {

// Type hash just like pointers.
template <> struct DenseMapInfo<mlir::pcc::PCCType> {
  using PCCType = ::mlir::pcc::PCCType;
  static PCCType getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return PCCType(static_cast<mlir::Type::ImplType *>(pointer));
  }
  static PCCType getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return PCCType(static_cast<mlir::Type::ImplType *>(pointer));
  }
  static unsigned getHashValue(PCCType val) { return mlir::hash_value(val); }
  static bool isEqual(PCCType LHS, PCCType RHS) { return LHS == RHS; }
};

} // namespace llvm
