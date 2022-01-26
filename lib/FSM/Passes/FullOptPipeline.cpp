#include "FSM/Passes/Passes.h"
#include "PassDetail.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::fsm;
namespace {
void pipelineBuilder(OpPassManager &pm) {
  pm.addPass(createStallingOptimizationPass());
  pm.addPass(createNonStallingOptimizationPass());
}
} // namespace

namespace mlir {
namespace fsm {
void registerFullOptimizationPipeline() {
  PassPipelineRegistration<>("fully-optimized",
                             "optimize both stalling and non stalling cases",
                             pipelineBuilder);
}
} // namespace fsm
} // namespace mlir