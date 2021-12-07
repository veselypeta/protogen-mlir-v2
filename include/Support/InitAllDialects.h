#pragma once

#include "mlir/IR/Dialect.h"
#include "PCC/PCCDialect.h"
#include "FSM/FSMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace mlir
{
    // add all mlir dialects to the registry
    inline void registerAllDialects(mlir::DialectRegistry &registry)
    {
        registry.insert<
            mlir::StandardOpsDialect,
            mlir::pcc::PCCDialect>();
    }

} // namespace mlir
