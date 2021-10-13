#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace pcc {

std::unique_ptr<OperationPass<ModuleOp>> createTransientStatesPass();

void initAllPCCPasses();

} // namespace pcc
} // namespace mlir