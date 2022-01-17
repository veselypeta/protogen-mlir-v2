#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace fsm {

std::unique_ptr<OperationPass<ModuleOp>> createStallingOptimizationPass();

void registerAllFSMPasses();

} // namespace fsm
} // namespace mlir