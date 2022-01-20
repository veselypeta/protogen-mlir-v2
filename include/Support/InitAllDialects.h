#pragma once

#include "FSM/FSMDialect.h"
#include "PCC/PCCDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dialect.h"

namespace mlir {
// add all mlir dialects to the registry
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::StandardOpsDialect, mlir::pcc::PCCDialect,
                  mlir::fsm::FSMDialect>();
}

} // namespace mlir
