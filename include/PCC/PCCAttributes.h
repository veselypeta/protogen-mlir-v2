#pragma once
#include "PCC/PCCDialect.h"
#include "PCC/PCCTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace pcc {}
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "PCC/PCCAttributes.h.inc"