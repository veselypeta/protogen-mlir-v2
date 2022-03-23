#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace fsm {

std::unique_ptr<OperationPass<ModuleOp>> createStallingOptimizationPass();
std::unique_ptr<OperationPass<ModuleOp>> createNonStallingOptimizationPass();
std::unique_ptr<OperationPass<ModuleOp>> createRemoveAwaitPass();

void registerAllFSMPasses();
void registerAllFSMPipelines();

} // namespace fsm
} // namespace mlir