#pragma once
#include "mlir/IR/Dialect.h"

namespace mlir {
namespace pcc {
class PCCType;
} // namespace pcc

} // namespace mlir

// pull in the dialect definition
#include "PCC/PCCDialect.h.inc"
