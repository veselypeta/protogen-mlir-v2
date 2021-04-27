#pragma once

#include "mlir/IR/Dialect.h"
#include "PCC/PCCDialect.h"

namespace mlir
{
    // add all mlir dialects to the registry
    inline void registerAllDialects(mlir::DialectRegistry &registry)
    {
        registry.insert<
            mlir::pcc::PCCDialect>();
    }

} // namespace mlir
