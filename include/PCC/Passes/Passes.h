#pragma once

#include "mlir/Pass/Pass.h"
#include "PCC/PCCOps.h"
#include <memory>

namespace mlir {
namespace pcc {

std::unique_ptr<OperationPass<ModuleOp>> createTransientStatesPass();
std::unique_ptr<OperationPass<ModuleOp>> createEndStatesPass();
std::unique_ptr<OperationPass<ModuleOp>> createMutexOpsPass();
std::unique_ptr<OperationPass<ModuleOp>> createStallingProtogenOptimizationPass();

void initAllPCCPasses();
void initAllPipelines();

} // namespace pcc
} // namespace mlir