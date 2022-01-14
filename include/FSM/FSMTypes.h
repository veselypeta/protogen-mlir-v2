#pragma once
#include "FSM/FSMDialect.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace fsm {
namespace detail {
// FWD Declare Type Storage instances here
}

// FWD Declares Types here
class IDType;
class DataType;
class MsgType;

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