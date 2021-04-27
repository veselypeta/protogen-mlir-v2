#pragma once
#include "PCC/PCCDialect.h"
#include "PCC/PCCOps.h"
#include "mlir/IR/Types.h"
#include <string>

namespace mlir {

namespace pcc {
namespace detail {
struct NetworkTypeStorage;
} // namespace detail

// Types
class IDType;
class NetworkType;

// this is a common base for all PCC Types
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

/// ID Type inherits from common base class PCCType
class IDType : public PCCType::TypeBase<IDType, PCCType, DefaultTypeStorage> {
public:
  using Base::Base;
  static IDType get(MLIRContext *context) { return Base::get(context); }
};

// Network Type inherits from common base class PCCType
class NetworkType : public PCCType::TypeBase<NetworkType, PCCType,
                                             detail::NetworkTypeStorage> {
public:
  using Base::Base;
  std::string getOrdering();
  typedef enum { ORDERED, UNORDERED } Ordering;
  static NetworkType get(MLIRContext *context, Ordering order);
};

} // namespace pcc
} // namespace mlir
