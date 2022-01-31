#pragma once
#include "FSM/FSMDialect.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace fsm {
namespace detail {
// FWD Declare Type Storage instances here
struct RangeTypeStorage;
struct SetTypeStorage;
} // namespace detail

// FWD Declares Types here
class IDType;
class DataType;
class MsgType;
class NetworkType;
class StateType;
class RangeType;
class SetType;

//===----------------------------------------------------------------------===//
// FSM Type
//===----------------------------------------------------------------------===//
class FSMType : public Type {
public:
  void print(llvm::raw_ostream &os) const;

  static bool classof(Type type) {
    return llvm::isa<mlir::fsm::FSMDialect>(type.getDialect());
  }

protected:
  using Type::Type;
};

//===----------------------------------------------------------------------===//
// ID Type
//===----------------------------------------------------------------------===//
/// ID Type inherits from common base class PCCType
class IDType : public FSMType::TypeBase<IDType, FSMType, DefaultTypeStorage> {
public:
  using Base::Base;
  static IDType get(MLIRContext *context) { return Base::get(context); }
};

//===----------------------------------------------------------------------===//
// Data Type
//===----------------------------------------------------------------------===//
class DataType
    : public FSMType::TypeBase<DataType, FSMType, DefaultTypeStorage> {
public:
  using Base::Base;
  static DataType get(MLIRContext *context) { return Base::get(context); }
};

//===----------------------------------------------------------------------===//
// Msg Type
//===----------------------------------------------------------------------===//
class MsgType : public FSMType::TypeBase<MsgType, FSMType, DefaultTypeStorage> {
public:
  using Base::Base;
  static MsgType get(MLIRContext *ctx) { return Base::get(ctx); }
};

//===----------------------------------------------------------------------===//
// State Type
//===----------------------------------------------------------------------===//
class StateType
    : public FSMType::TypeBase<StateType, FSMType, DefaultTypeStorage> {
public:
  using Base::Base;
  static StateType get(MLIRContext *ctx) { return Base::get(ctx); }
};

//===----------------------------------------------------------------------===//
// Network Type
//===----------------------------------------------------------------------===//
class NetworkType
    : public FSMType::TypeBase<NetworkType, FSMType, DefaultTypeStorage> {
public:
  using Base::Base;
  static NetworkType get(MLIRContext *ctx) { return Base::get(ctx); }
};

//===----------------------------------------------------------------------===//
// Range Type
//===----------------------------------------------------------------------===//

class RangeType
    : public FSMType::TypeBase<RangeType, FSMType, detail::RangeTypeStorage> {
public:
  using Base::Base;
  static RangeType get(MLIRContext *ctx, size_t start, size_t end);
  size_t getStart();
  size_t getEnd();
};

//===----------------------------------------------------------------------===//
// Set Type
//===----------------------------------------------------------------------===//

class SetType
    : public FSMType::TypeBase<SetType, FSMType, detail::SetTypeStorage> {
public:
  using Base::Base;
  Type getElementType();
  size_t getNumElements();
  static SetType get(Type elemType, size_t count);
};

/// used for verification
LogicalResult areTypesCompatible(Type t1, Type t2);
} // namespace fsm
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "FSM/FSMTypes.h.inc"

namespace llvm {

// Type hash just like pointers.
template <> struct DenseMapInfo<mlir::fsm::FSMType> {
  using FSMType = ::mlir::fsm::FSMType;
  static FSMType getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return FSMType(static_cast<mlir::Type::ImplType *>(pointer));
  }
  static FSMType getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return FSMType(static_cast<mlir::Type::ImplType *>(pointer));
  }
  static unsigned getHashValue(FSMType val) { return mlir::hash_value(val); }
  static bool isEqual(FSMType LHS, FSMType RHS) { return LHS == RHS; }
};

} // namespace llvm